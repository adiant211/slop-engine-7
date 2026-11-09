// SmolLM2-135M-Instruct CPU inference engine.
// int8 (per-output-row) weight-only quantisation, f32 activations, AVX2/FMA kernels, 3-thread pool.
use std::arch::x86_64::*;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static STATS: [AtomicU64; 8] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];
#[inline]
fn acc(slot: usize, t: Instant) {
    STATS[slot].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
}
fn dump_stats() {
    if std::env::var("SMOL_TIME").is_ok() {
        let names = ["rmsnorm", "qkv", "rope+kv", "attn", "oproj", "mlp", "logits", "x"];
        let mut s = String::new();
        for i in 0..8 {
            s += &format!("{}={:.2} ", names[i], STATS[i].load(Ordering::Relaxed) as f64 / 1e6);
        }
        eprintln!("STATS {}", s);
    }
}

const H: usize = 576;
const NH: usize = 9;
const NKV: usize = 3;
const HD: usize = 64;
const FF: usize = 1536;
const VV: usize = 49152;
const NL: usize = 30;
const EPS: f32 = 1e-5;
const QDIM: usize = NH * HD;
const KVDIM: usize = NKV * HD;
const QKV: usize = QDIM + 2 * KVDIM;
const GU: usize = 2 * FF;
const MAXPOS: usize = 8192;
const CB: usize = 512; // prefill chunk

// ---------------------------------------------------------------- thread pool
#[derive(Clone, Copy)]
struct P<T>(*const T);
unsafe impl<T> Send for P<T> {}
unsafe impl<T> Sync for P<T> {}
impl<T> P<T> {
    #[inline(always)]
    fn get(&self) -> *const T {
        self.0
    }
}
#[derive(Clone, Copy)]
struct PM<T>(*mut T);
unsafe impl<T> Send for PM<T> {}
unsafe impl<T> Sync for PM<T> {}
impl<T> PM<T> {
    #[inline(always)]
    fn get(&self) -> *mut T {
        self.0
    }
}


struct St {
    slot: std::cell::UnsafeCell<[usize; 2]>,
    epoch: AtomicU64,
    done: std::sync::atomic::AtomicUsize,
}
unsafe impl Sync for St {}
unsafe impl Send for St {}

struct Pool {
    n: usize,
    st: Arc<St>,
}
impl Pool {
    fn new(n: usize) -> Pool {
        let st = Arc::new(St {
            slot: std::cell::UnsafeCell::new([0usize; 2]),
            epoch: AtomicU64::new(0),
            done: std::sync::atomic::AtomicUsize::new(0),
        });
        for tid in 1..n {
            let s = st.clone();
            std::thread::spawn(move || {
                let mut my: u64 = 0;
                loop {
                    let mut spins = 0u32;
                    while s.epoch.load(Ordering::Relaxed) == my {
                        std::hint::spin_loop();
                        spins += 1;
                        if spins > 20_000 {
                            std::thread::sleep(std::time::Duration::from_micros(200));
                            spins = 0;
                        }
                    }
                    my = s.epoch.load(Ordering::Acquire);
                    let f: &'static (dyn Fn(usize) + Sync) =
                        unsafe { std::mem::transmute::<[usize; 2], &(dyn Fn(usize) + Sync)>(*s.slot.get()) };
                    f(tid);
                    s.done.fetch_add(1, Ordering::Release);
                }
            });
        }
        Pool { n, st }
    }
    #[inline]
    fn run<F: Fn(usize) + Sync>(&self, f: F) {
        if self.n == 1 {
            f(0);
            return;
        }
        unsafe {
            *self.st.slot.get() = std::mem::transmute::<&(dyn Fn(usize) + Sync), [usize; 2]>(&f);
        }
        self.st.done.store(0, Ordering::Relaxed);
        self.st.epoch.fetch_add(1, Ordering::Release);
        f(0);
        while self.st.done.load(Ordering::Acquire) < self.n - 1 {
            std::hint::spin_loop();
        }
    }
}

#[inline]
fn split(total: usize, n: usize, tid: usize) -> (usize, usize) {
    if n == 1 {
        return (0, total);
    }
    let base = total / n;
    let rem = total % n;
    let lo = tid * base + if tid < rem { tid } else { rem };
    let hi = lo + base + if tid < rem { 1 } else { 0 };
    (lo, hi)
}

// ---------------------------------------------------------------- kernels
#[inline(always)]
unsafe fn hsum(a: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(a);
    let hi = _mm256_extractf128_ps::<1>(a);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

#[inline(always)]
unsafe fn hsum_i32(a: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(a);
    let hi = _mm256_extracti128_si256::<1>(a);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
    _mm_cvtsi128_si32(s)
}

// weight matrix: int8 (per-output-row scale) or int4 (per-32-block scale, offset binary +8)
struct Qw {
    n: usize,
    k: usize,
    g4: bool,
    q: Vec<u8>,   // int8: n*k bytes (i8 bits) ; int4: n*k/2 (byte i: lo=w[b*32+i], hi=w[b*32+16+i])
    sc: Vec<f32>, // int8: n rows ; int4: n*(k/32) blocks
}

// one output row, MM input rows at once. x rows are int8-quantised i16, xs = per-row scales.
#[inline(always)]
unsafe fn dot8_mm<const MM: usize>(
    wq: *const u8,
    sc: f32,
    xs: *const f32,
    k: usize,
    xb: *const i16,
    o: *mut f32,
    stride: usize,
) {
    let nb = k / 32;
    let mut ac = [_mm256_setzero_ps(); 4];
    let mut b = 0usize;
    while b < nb {
        let sp = xs.add(b);
        for r in 0..MM {
            let p = blk8(wq, xb.add(r * k), b);
            ac[r] = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(p),
                _mm256_set1_ps(sc * *sp.add(r * nb)),
                ac[r],
            );
        }
        b += 1;
    }
    for r in 0..MM {
        *o.add(r * stride) = hsum(ac[r]);
    }
}

#[allow(dead_code)]
#[inline(always)]
unsafe fn dot4_mm<const MM: usize>(
    wq: *const u8,
    sc4: *const f32,
    xs: *const f32,
    k: usize,
    xb: *const i16,
    o: *mut f32,
    stride: usize,
) {
    let nb = k / 32;
    let mut ac = [_mm256_setzero_ps(); 4];
    let eight = _mm256_set1_epi8(8);
    let m0f = _mm_set1_epi8(0x0f);
    let mut b = 0usize;
    while b < nb {
        let v = _mm_loadu_si128(wq.add(b * 16) as *const __m128i);
        let lo = _mm_and_si128(v, m0f);
        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), m0f);
        let wv = _mm256_sub_epi8(
            _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi),
            eight,
        );
        let wl = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(wv));
        let wh = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(wv));
        let s = _mm256_set1_ps(*sc4.add(b));
        for r in 0..MM {
            let xp = xb.add(r * k + b * 32);
            let p = _mm256_add_epi32(
                _mm256_madd_epi16(_mm256_loadu_si256(xp as *const __m256i), wl),
                _mm256_madd_epi16(_mm256_loadu_si256(xp.add(16) as *const __m256i), wh),
            );
            ac[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), s, ac[r]);
        }
        b += 1;
    }
    for r in 0..MM {
        *o.add(r * stride) = hsum(ac[r]) * *xs.add(r);
    }
}

#[inline(always)]
unsafe fn blk8(wq: *const u8, xb: *const i16, b: usize) -> __m256i {
    let wl = _mm256_cvtepi8_epi16(_mm_loadu_si128(wq.add(b * 32) as *const __m128i));
    let wh = _mm256_cvtepi8_epi16(_mm_loadu_si128(wq.add(b * 32 + 16) as *const __m128i));
    _mm256_add_epi32(
        _mm256_madd_epi16(_mm256_loadu_si256(xb.add(b * 32) as *const __m256i), wl),
        _mm256_madd_epi16(_mm256_loadu_si256(xb.add(b * 32 + 16) as *const __m256i), wh),
    )
}

// int8 weights, per-32-block activation scales, one output row, 4 f32 chains
#[inline(always)]
unsafe fn dot8_1(wq: *const u8, wsc: f32, asc: *const f32, k: usize, xb: *const i16, o: *mut f32) {
    let nb = k / 32;
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let mut b = 0usize;
    while b + 4 <= nb {
        let p0 = blk8(wq, xb, b);
        let p1 = blk8(wq, xb, b + 1);
        let p2 = blk8(wq, xb, b + 2);
        let p3 = blk8(wq, xb, b + 3);
        a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(wsc * *asc.add(b)), a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(wsc * *asc.add(b + 1)), a1);
        a2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p2), _mm256_set1_ps(wsc * *asc.add(b + 2)), a2);
        a3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p3), _mm256_set1_ps(wsc * *asc.add(b + 3)), a3);
        b += 4;
    }
    while b < nb {
        let p0 = blk8(wq, xb, b);
        a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(wsc * *asc.add(b)), a0);
        b += 1;
    }
    *o = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
}

#[inline(always)]
unsafe fn row_mm<const MM: usize>(
    w: &Qw,
    j: usize,
    k: usize,
    xb: *const i16,
    sb: *const f32,
    o: *mut f32,
    stride: usize,
) {
    let nb = k / 32;
    let _ = nb;
    if MM == 1 {
        dot8_1(
            w.q.as_ptr().add(j * k),
            *w.sc.get_unchecked(j),
            sb,
            k,
            xb,
            o,
        );
        return;
    }
    dot8_mm::<MM>(
        w.q.as_ptr().add(j * k),
        *w.sc.get_unchecked(j),
        sb,
        k,
        xb,
        o,
        stride,
    );
}

// xq: m*k int8 activations (i16), xs: m scales. out: m*n, row stride n.
fn matmul(pool: &Pool, xq: &[i16], xs: &[f32], m: usize, w: &Qw, out: &mut [f32]) {
    let n = w.n;
    let k = w.k;
    debug_assert!(xq.len() >= m * k && xs.len() >= m && out.len() >= m * n);
    let nt = pool.n;
    let pw = P(w as *const Qw);
    let px = P(xq.as_ptr());
    let ps = P(xs.as_ptr());
    let po = PM(out.as_mut_ptr());
    pool.run(move |tid| unsafe {
        let w = &*pw.get();
        let (lo, hi) = split(n, nt, tid);
        if lo >= hi {
            return;
        }
        let mut i0 = 0usize;
        while i0 < m {
            let mm = core::cmp::min(4, m - i0);
            let xb = px.get().add(i0 * k);
            let sb = ps.get().add(i0 * (k / 32));
            let ob = po.get().add(i0 * n);
            let mut j = lo;
            while j < hi {
                let o = ob.add(j);
                match mm {
                    1 => row_mm::<1>(w, j, k, xb, sb, o, 0),
                    2 => row_mm::<2>(w, j, k, xb, sb, o, n),
                    3 => row_mm::<3>(w, j, k, xb, sb, o, n),
                    _ => row_mm::<4>(w, j, k, xb, sb, o, n),
                }
                j += 1;
            }
            i0 += mm;
        }
    });
}

// quantise m rows of f32 (k each) to int8 with per-32-block scales; result in i16 lanes
fn quant_act(x: &[f32], m: usize, k: usize, xq: &mut [i16], xs: &mut [f32]) {
    let nb = k / 32;
    for r in 0..m {
        let src = &x[r * k..(r + 1) * k];
        let dst = &mut xq[r * k..(r + 1) * k];
        let srow = &mut xs[r * nb..(r + 1) * nb];
        for b in 0..nb {
            let blk = &src[b * 32..b * 32 + 32];
            let mut mx = 0f32;
            let mut i = 0usize;
            unsafe {
                let mut a = _mm256_setzero_ps();
                let mut c = _mm256_setzero_ps();
                while i + 16 <= 32 {
                    let v0 = _mm256_loadu_ps(blk.as_ptr().add(i));
                    let v1 = _mm256_loadu_ps(blk.as_ptr().add(i + 8));
                    a = _mm256_max_ps(a, _mm256_max_ps(v0, v1));
                    c = _mm256_min_ps(c, _mm256_min_ps(v0, v1));
                    i += 16;
                }
                mx = hsum_max(a).max(-hsum_min(c));
            }
            let sc = if mx > 0.0 { mx / 127.0 } else { 1.0 };
            let inv = 1.0 / sc;
            srow[b] = sc;
            let d = &mut dst[b * 32..b * 32 + 32];
            let mut i = 0usize;
            unsafe {
                let vi = _mm256_set1_ps(inv);
                while i + 8 <= 32 {
                    let v = _mm256_mul_ps(_mm256_loadu_ps(blk.as_ptr().add(i)), vi);
                    let r8 = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    let n8 = _mm256_cvtps_epi32(r8);
                    let l = _mm256_castsi256_si128(n8);
                    let h = _mm256_extracti128_si256::<1>(n8);
                    let pp = _mm_packs_epi32(l, h);
                    _mm_storeu_si128(d.as_mut_ptr().add(i) as *mut __m128i, pp);
                    i += 8;
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn exp256(x: __m256) -> __m256 {
    let hi = _mm256_set1_ps(87.0);
    let lo = _mm256_set1_ps(-87.0);
    let x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
    let log2e = _mm256_set1_ps(1.4426950408889634);
    let n = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let r = _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(0.6931471805599453)));
    // exp(r) horner deg 6
    let mut p = _mm256_set1_ps(1.0 / 720.0);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 120.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 24.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 6.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
    let ni = _mm256_cvtps_epi32(n);
    let ex = _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23);
    _mm256_mul_ps(p, _mm256_castsi256_ps(ex))
}

fn silu_mul(g: &mut [f32], u: &[f32]) {
    let n = g.len();
    let mut i = 0;
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let neg = _mm256_set1_ps(-1.0);
        while i + 8 <= n {
            let x = _mm256_loadu_ps(g.as_ptr().add(i));
            let e = exp256(_mm256_mul_ps(x, neg));
            let d = _mm256_add_ps(one, e);
            let s = _mm256_div_ps(x, d);
            let uv = _mm256_loadu_ps(u.as_ptr().add(i));
            _mm256_storeu_ps(g.as_mut_ptr().add(i), _mm256_mul_ps(s, uv));
            i += 8;
        }
    }
    while i < n {
        let x = g[i];
        g[i] = (x / (1.0 + (-x).exp())) * u[i];
        i += 1;
    }
}

fn rmsnorm(h: &[f32], m: usize, w: &[f32], out: &mut [f32]) {
    let k = w.len();
    for r in 0..m {
        let src = &h[r * k..(r + 1) * k];
        let dst = &mut out[r * k..(r + 1) * k];
        let mut s;
        let mut i = 0;
        unsafe {
            let mut a = _mm256_setzero_ps();
            while i + 8 <= k {
                let v = _mm256_loadu_ps(src.as_ptr().add(i));
                a = _mm256_fmadd_ps(v, v, a);
                i += 8;
            }
            s = hsum(a);
        }
        while i < k {
            s += src[i] * src[i];
            i += 1;
        }
        let scale = 1.0 / ((s / (k as f32)) + EPS).sqrt();
        let mut i = 0;
        unsafe {
            let sv = _mm256_set1_ps(scale);
            while i + 8 <= k {
                let v = _mm256_loadu_ps(src.as_ptr().add(i));
                let wv = _mm256_loadu_ps(w.as_ptr().add(i));
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_mul_ps(v, sv), wv));
                i += 8;
            }
        }
        while i < k {
            dst[i] = src[i] * scale * w[i];
            i += 1;
        }
    }
}

// ---------------------------------------------------------------- model
struct Layer {
    wqkv: Qw,
    wo: Qw,
    wgu: Qw,
    wdown: Qw,
    an: Vec<f32>,
    fnn: Vec<f32>,
}

struct Model {
    layers: Vec<Layer>,
    embed: Qw,
    norm: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

#[inline(always)]
fn bf2f(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn quantize8(data: &[u8], n: usize, k: usize, pool: &Pool) -> Qw {
    let mut q = vec![0u8; n * k];
    let mut sc = vec![0f32; n];
    let nt = pool.n;
    let pq = PM(q.as_mut_ptr());
    let ps = PM(sc.as_mut_ptr());
    let pb = P(data.as_ptr());
    pool.run(move |tid| unsafe {
        let (lo, hi) = split(n, nt, tid);
        for r in lo..hi {
            let src = pb.get().add(r * k * 2) as *const u16;
            let mut mabs = 0f32;
            for j in 0..k {
                let a = bf2f(*src.add(j)).abs();
                if a > mabs {
                    mabs = a;
                }
            }
            let s = if mabs > 0.0 { mabs / 127.0 } else { 1.0 };
            let inv = 1.0 / s;
            let dst = pq.get().add(r * k);
            for j in 0..k {
                let v = bf2f(*src.add(j)) * inv;
                let iv = (v.round() as i32).clamp(-127, 127);
                *dst.add(j) = iv as i8 as u8;
            }
            *ps.get().add(r) = s;
        }
    });
    Qw { n, k, g4: false, q, sc }
}

fn quantize4(data: &[u8], n: usize, k: usize, pool: &Pool) -> Qw {
    assert!(k % 32 == 0);
    let nb = k / 32;
    let mut q = vec![0u8; n * (k / 2)];
    let mut sc = vec![0f32; n * nb];
    let nt = pool.n;
    let pq = PM(q.as_mut_ptr());
    let ps = PM(sc.as_mut_ptr());
    let pb = P(data.as_ptr());
    pool.run(move |tid| unsafe {
        let (lo, hi) = split(n, nt, tid);
        for r in lo..hi {
            let src = pb.get().add(r * k * 2) as *const u16;
            let dst = pq.get().add(r * (k / 2));
            for b in 0..nb {
                let mut mabs = 0f32;
                for i in 0..32 {
                    let a = bf2f(*src.add(b * 32 + i)).abs();
                    if a > mabs {
                        mabs = a;
                    }
                }
                let s = if mabs > 0.0 { mabs / 7.0 } else { 1.0 };
                let inv = 1.0 / s;
                *ps.get().add(r * nb + b) = s;
                for i in 0..16 {
                    let v0 = bf2f(*src.add(b * 32 + i)) * inv;
                    let v1 = bf2f(*src.add(b * 32 + 16 + i)) * inv;
                    let q0 = ((v0.round() as i32) + 8).clamp(0, 15) as u8;
                    let q1 = ((v1.round() as i32) + 8).clamp(0, 15) as u8;
                    *dst.add(b * 16 + i) = q0 | (q1 << 4);
                }
            }
        }
    });
    Qw { n, k, g4: true, q, sc }
}

fn f32_vec(data: &[u8], n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    let p = data.as_ptr() as *const u16;
    for i in 0..n {
        v.push(bf2f(unsafe { *p.add(i) }));
    }
    v
}

fn merge_rows(parts: &[&Qw]) -> Qw {
    let k = parts[0].k;
    let n: usize = parts.iter().map(|p| p.n).sum();
    let mut q = Vec::with_capacity(n * k);
    let mut sc = Vec::with_capacity(n);
    for p in parts {
        assert_eq!(p.k, k);
        q.extend_from_slice(&p.q);
        sc.extend_from_slice(&p.sc);
    }
    Qw { n, k, g4: false, q, sc }
}

fn load_model(dir: &str, pool: &Pool) -> Model {
    let path = format!("{}/model.safetensors", dir);
    let data = std::fs::read(&path).expect("cannot read model.safetensors");
    let st = safetensors::SafeTensors::deserialize(&data).expect("bad safetensors");
    macro_rules! t {
        ($n:expr) => {
            st.tensor($n).unwrap_or_else(|e| panic!("tensor {} {:?}", $n, e))
        };
    }
    macro_rules! q2 {
        ($n:expr, $nn:expr, $kk:expr) => {{
            let t = t!($n);
            let sh = t.shape();
            assert_eq!((sh[0], sh[1]), ($nn, $kk), "{} shape", $n);
            quantize8(t.data(), $nn, $kk, pool)
        }};
    }
    macro_rules! q4 {
        ($n:expr, $nn:expr, $kk:expr) => {{
            let t = t!($n);
            let sh = t.shape();
            assert_eq!((sh[0], sh[1]), ($nn, $kk), "{} shape", $n);
            quantize4(t.data(), $nn, $kk, pool)
        }};
    }
    let embed = q2!("model.embed_tokens.weight", VV, H);
    let nrm = {
        let t = t!("model.norm.weight");
        f32_vec(t.data(), H)
    };
    let mut layers = Vec::with_capacity(NL);
    for l in 0..NL {
        let p = format!("model.layers.{}", l);
        let wq = q2!(format!("{}.self_attn.q_proj.weight", p).as_str(), QDIM, H);
        let wk = q2!(format!("{}.self_attn.k_proj.weight", p).as_str(), KVDIM, H);
        let wv = q2!(format!("{}.self_attn.v_proj.weight", p).as_str(), KVDIM, H);
        let wo = q2!(format!("{}.self_attn.o_proj.weight", p).as_str(), QDIM, H);
        let wgate = q2!(format!("{}.mlp.gate_proj.weight", p).as_str(), FF, H);
        let wup = q2!(format!("{}.mlp.up_proj.weight", p).as_str(), FF, H);
        let wdown = q2!(format!("{}.mlp.down_proj.weight", p).as_str(), H, FF);
        let an = {
            let t = t!(format!("{}.input_layernorm.weight", p).as_str());
            f32_vec(t.data(), H)
        };
        let fnn = {
            let t = t!(format!("{}.post_attention_layernorm.weight", p).as_str());
            f32_vec(t.data(), H)
        };
        let wqkv = merge_rows(&[&wq, &wk, &wv]);
        let wgu = merge_rows(&[&wgate, &wup]);
        drop((wq, wk, wv, wgate, wup));
        layers.push(Layer { wqkv, wo, wgu, wdown, an, fnn });
    }
    // rope tables
    let mut cos = vec![0f32; MAXPOS * 32];
    let mut sin = vec![0f32; MAXPOS * 32];
    for i in 0..32 {
        let inv = (1.0f64 / (100000.0f64).powf((2 * i) as f64 / 64.0)) as f32;
        for p in 0..MAXPOS {
            let a = p as f32 * inv;
            cos[p * 32 + i] = a.cos();
            sin[p * 32 + i] = a.sin();
        }
    }
    Model { layers, embed, norm: nrm, cos, sin }
}

// ---------------------------------------------------------------- state
struct Caches {
    ctx: usize,
    kc: Vec<f32>, // [NL*NKV][64][ctx]
    vc: Vec<f32>, // [NL*NKV][ctx][64]
    len: usize,
}

impl Caches {
    fn new(ctx: usize) -> Caches {
        Caches {
            ctx,
            kc: vec![0f32; NL * NKV * 64 * ctx],
            vc: vec![0f32; NL * NKV * ctx * 64],
            len: 0,
        }
    }
}

struct Bufs {
    h: Vec<f32>,
    x: Vec<f32>,
    tmp: Vec<f32>,
    qkv: Vec<f32>,
    ao: Vec<f32>,
    g: Vec<f32>,
    gu: Vec<f32>,
    xq: Vec<i16>,
    xs: Vec<f32>,
    gq: Vec<i16>,
    gs: Vec<f32>,
    aq: Vec<i16>,
    az: Vec<f32>,
}

impl Bufs {
    fn new(m: usize) -> Bufs {
        Bufs {
            h: vec![0f32; m * H],
            x: vec![0f32; m * H],
            tmp: vec![0f32; m * H],
            qkv: vec![0f32; m * QKV],
            ao: vec![0f32; m * QDIM],
            g: vec![0f32; m * FF],
            gu: vec![0f32; m * GU],
            xq: vec![0i16; m * H],
            xs: vec![0f32; m * (H / 32) + 32],
            gq: vec![0i16; m * FF],
            gs: vec![0f32; m * (FF / 32) + 32],
            aq: vec![0i16; m * QDIM],
            az: vec![0f32; m * (QDIM / 32) + 32],
        }
    }
}

fn rope(v: &mut [f32], m: usize, pos0: usize, nh: usize, stride: usize, base: usize, cos: &[f32], sin: &[f32]) {
    for i in 0..m {
        let p = pos0 + i;
        let cs = &cos[p * 32..p * 32 + 32];
        let sn = &sin[p * 32..p * 32 + 32];
        for hh in 0..nh {
            let off = i * stride + base + hh * 64;
            for d in 0..32 {
                let a = v[off + d];
                let b = v[off + d + 32];
                v[off + d] = a * cs[d] - b * sn[d];
                v[off + d + 32] = b * cs[d] + a * sn[d];
            }
        }
    }
}

fn write_kv(nc: &mut Caches, l: usize, m: usize, kb: &[f32], pos0: usize) {
    let ctx = nc.ctx;
    for kk in 0..NKV {
        let kbase = (l * NKV + kk) * 64 * ctx;
        let vbase = (l * NKV + kk) * ctx * 64;
        for i in 0..m {
            let p = pos0 + i;
            for d in 0..64 {
                nc.kc[kbase + d * ctx + p] = kb[i * QKV + QDIM + kk * 64 + d];
                nc.vc[vbase + p * 64 + d] = kb[i * QKV + QDIM + KVDIM + kk * 64 + d];
            }
        }
    }
}

unsafe fn mask8(n: usize) -> (__m256, __m256i) {
    // mask for n valid lanes (n<=8)
    let mut mask = [0i32; 8];
    for i in 0..n {
        mask[i] = -1;
    }
    (
        _mm256_castsi256_ps(_mm256_loadu_si256(mask.as_ptr() as *const __m256i)),
        _mm256_loadu_si256(mask.as_ptr() as *const __m256i),
    )
}

fn attention(
    l: usize,
    nc: &mut Caches,
    q: &[f32],
    ao: &mut [f32],
    m: usize,
    pos0: usize,
    pool: &Pool,
) {
    let ctx = nc.ctx;
    let pk = P(nc.kc.as_ptr());
    let pv = P(nc.vc.as_ptr());
    let pq = P(q.as_ptr());
    let pa = PM(ao.as_mut_ptr());
    let nt = pool.n;
    let br = if m <= 8 { 1 } else { 64 };
    let nblk = (m + br - 1) / br;
    let tasks = NH * nblk;
    pool.run(move |tid| {
        let mut t = tid;
        while t < tasks {
            let h = t / nblk;
            let b = t % nblk;
            let i0 = b * br;
            let rows = core::cmp::min(br, m - i0);
            let kvh = h / (NH / NKV);
            unsafe {
                let kbase = pk.get().add((l * NKV + kvh) * 64 * ctx);
                let vbase = pv.get().add((l * NKV + kvh) * ctx * 64);
                let total = pos0 + i0 + rows;
                let mut sc = vec![0f32; rows * total];
                // ---- scores
                let mut r0 = 0;
                while r0 < rows {
                    let rr = core::cmp::min(4, rows - r0);
                    let qb = pq.get().add((i0 + r0) * QKV + h * 64);
                    let sb = sc.as_mut_ptr().add(r0 * total);
                    let mut j0 = 0;
                    while j0 < total {
                        let jn = core::cmp::min(8, total - j0);
                        let (mv, mi) = mask8(jn);
                        let mut a0 = _mm256_setzero_ps();
                        let mut a1 = _mm256_setzero_ps();
                        let mut a2 = _mm256_setzero_ps();
                        let mut a3 = _mm256_setzero_ps();
                        for d in 0..64 {
                            let kvv = _mm256_maskload_ps(kbase.add(d * ctx + j0), mi);
                            a0 = _mm256_fmadd_ps(_mm256_set1_ps(*qb.add(d)), kvv, a0);
                            if rr > 1 {
                                a1 = _mm256_fmadd_ps(_mm256_set1_ps(*qb.add(QKV + d)), kvv, a1);
                            }
                            if rr > 2 {
                                a2 = _mm256_fmadd_ps(_mm256_set1_ps(*qb.add(2 * QKV + d)), kvv, a2);
                            }
                            if rr > 3 {
                                a3 = _mm256_fmadd_ps(_mm256_set1_ps(*qb.add(3 * QKV + d)), kvv, a3);
                            }
                        }
                        let s = _mm256_set1_ps(0.125);
                        _mm256_maskstore_ps(sb.add(j0), mi, _mm256_mul_ps(a0, s));
                        if rr > 1 {
                            _mm256_maskstore_ps(sb.add(total + j0), mi, _mm256_mul_ps(a1, s));
                        }
                        if rr > 2 {
                            _mm256_maskstore_ps(sb.add(2 * total + j0), mi, _mm256_mul_ps(a2, s));
                        }
                        if rr > 3 {
                            _mm256_maskstore_ps(sb.add(3 * total + j0), mi, _mm256_mul_ps(a3, s));
                        }
                        let _ = mv;
                        j0 += 8;
                    }
                    r0 += rr;
                }
                // ---- softmax + weighted sum of V (4 rows at a time)
                let mut r = 0;
                while r < rows {
                    let rr = core::cmp::min(4, rows - r);
                    let maxj = pos0 + i0 + r;
                    for x in 0..rr {
                        let len_r = maxj + 1 + x;
                        let sr = sc.as_mut_ptr().add((r + x) * total);
                        let mut mx = f32::NEG_INFINITY;
                        let mut i = 0;
                        let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
                        while i + 8 <= len_r {
                            let v = _mm256_loadu_ps(sr.add(i));
                            acc = _mm256_max_ps(acc, v);
                            i += 8;
                        }
                        let mut hm = hsum_max(acc);
                        while i < len_r {
                            if *sr.add(i) > hm {
                                hm = *sr.add(i);
                            }
                            i += 1;
                        }
                        mx = hm;
                        // exp
                        let mut sum = 0f32;
                        let mut sv = _mm256_setzero_ps();
                        let mut i = 0;
                        let mvv = _mm256_set1_ps(mx);
                        while i + 8 <= len_r {
                            let e = exp256(_mm256_sub_ps(_mm256_loadu_ps(sr.add(i)), mvv));
                            _mm256_storeu_ps(sr.add(i), e);
                            sv = _mm256_add_ps(sv, e);
                            i += 8;
                        }
                        sum += hsum(sv);
                        while i < len_r {
                            let e = (*sr.add(i) - mx).exp();
                            *sr.add(i) = e;
                            sum += e;
                            i += 1;
                        }
                        let inv = 1.0 / sum;
                        let iv = _mm256_set1_ps(inv);
                        let mut i = 0;
                        while i + 8 <= len_r {
                            let e = _mm256_loadu_ps(sr.add(i));
                            _mm256_storeu_ps(sr.add(i), _mm256_mul_ps(e, iv));
                            i += 8;
                        }
                        while i < len_r {
                            *sr.add(i) *= inv;
                            i += 1;
                        }
                    }
                    for x in 0..rr {
                        let len_r = maxj + 1 + x;
                        let sr = sc.as_ptr().add((r + x) * total);
                        let mut acc = [_mm256_setzero_ps(); 8];
                        let mut j = 0;
                        while j < len_r {
                            let p = _mm256_set1_ps(*sr.add(j));
                            let vp = vbase.add(j * 64);
                            acc[0] = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp), acc[0]);
                            acc[1] = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp.add(8)), acc[1]);
                            acc[2] = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp.add(16)), acc[2]);
                            acc[3] = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp.add(24)), acc[3]);
                            acc[4] = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp.add(32)), acc[4]);
                            acc[5] = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp.add(40)), acc[5]);
                            acc[6] = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp.add(48)), acc[6]);
                            acc[7] = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp.add(56)), acc[7]);
                            j += 1;
                        }
                        let op = pa.get().add((i0 + r + x) * QDIM + h * 64);
                        for t2 in 0..8 {
                            _mm256_storeu_ps(op.add(t2 * 8), acc[t2]);
                        }
                    }
                    r += rr;
                }
            }
            t += nt;
        }
    });
}

#[inline(always)]
unsafe fn hsum_min(a: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(a);
    let hi = _mm256_extractf128_ps::<1>(a);
    let s = _mm_min_ps(lo, hi);
    let s = _mm_min_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_min_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

#[inline(always)]
unsafe fn hsum_max(a: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(a);
    let hi = _mm256_extractf128_ps::<1>(a);
    let s = _mm_max_ps(lo, hi);
    let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_max_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

fn forward_block(
    model: &Model,
    pool: &Pool,
    nc: &mut Caches,
    b: &mut Bufs,
    m: usize,
) {
    let pos0 = nc.len;
    for l in 0..NL {
        let lay = &model.layers[l];
        let mut t = Instant::now();
        rmsnorm(&b.h, m, &lay.an, &mut b.x);
        quant_act(&b.x, m, H, &mut b.xq, &mut b.xs);
        acc(0, t);
        t = Instant::now();
        matmul(pool, &b.xq, &b.xs, m, &lay.wqkv, &mut b.qkv);
        acc(1, t);
        t = Instant::now();
        rope(&mut b.qkv, m, pos0, NH, QKV, 0, &model.cos, &model.sin);
        rope(&mut b.qkv, m, pos0, NKV, QKV, QDIM, &model.cos, &model.sin);
        write_kv(nc, l, m, &b.qkv, pos0);
        acc(2, t);
        t = Instant::now();
        attention(l, nc, &b.qkv, &mut b.ao, m, pos0, pool);
        acc(3, t);
        t = Instant::now();
        quant_act(&b.ao, m, QDIM, &mut b.aq, &mut b.az);
        matmul(pool, &b.aq, &b.az, m, &lay.wo, &mut b.tmp);
        for i in 0..m * H {
            b.h[i] += b.tmp[i];
        }
        acc(4, t);
        t = Instant::now();
        rmsnorm(&b.h, m, &lay.fnn, &mut b.x);
        quant_act(&b.x, m, H, &mut b.xq, &mut b.xs);
        matmul(pool, &b.xq, &b.xs, m, &lay.wgu, &mut b.gu);
        for r in 0..m {
            let (gpart, upart) = b.gu[r * GU..(r + 1) * GU].split_at_mut(FF);
            b.g[r * FF..(r + 1) * FF].copy_from_slice(gpart);
            silu_mul(&mut b.g[r * FF..(r + 1) * FF], upart);
        }
        quant_act(&b.g, m, FF, &mut b.gq, &mut b.gs);
        matmul(pool, &b.gq, &b.gs, m, &lay.wdown, &mut b.tmp);
        acc(5, t);
        for i in 0..m * H {
            b.h[i] += b.tmp[i];
        }
    }
    rmsnorm(&b.h, m, &model.norm, &mut b.x);
    b.h[..m * H].copy_from_slice(&b.x[..m * H]);
    nc.len += m;
}

fn embed_rows(model: &Model, ids: &[u32], h: &mut [f32]) {
    let k = H;
    for (i, &id) in ids.iter().enumerate() {
        let r = id as usize;
        let q = &model.embed.q[r * k..(r + 1) * k];
        let s = model.embed.sc[r];
        let dst = &mut h[i * k..(i + 1) * k];
        for j in 0..k {
            dst[j] = (q[j] as i8) as f32 * s;
        }
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi as u32
}

fn write_f32s(f: &mut std::fs::File, v: &[f32]) {
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    f.write_all(bytes).unwrap();
}

fn write_u32s(f: &mut std::fs::File, v: &[u32]) {
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    f.write_all(bytes).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: smol ppl|gen ...");
        std::process::exit(2);
    }
    let mode = args[0].clone();
    let mut model_dir = String::from("model");
    let mut tokens_in: Option<String> = None;
    let mut ctx: usize = 1024;
    let mut logits_out: Option<String> = None;
    let mut prompt_file: Option<String> = None;
    let mut chat = false;
    let mut n_gen: Option<usize> = None;
    let mut force: Option<String> = None;
    let mut tokens_out: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_dir = args[i + 1].clone();
                i += 2
            }
            "--tokens" => {
                tokens_in = Some(args[i + 1].clone());
                i += 2
            }
            "--ctx" => {
                ctx = args[i + 1].parse().unwrap();
                i += 2
            }
            "--logits" => {
                logits_out = Some(args[i + 1].clone());
                i += 2
            }
            "--prompt-file" => {
                prompt_file = Some(args[i + 1].clone());
                i += 2
            }
            "--chat" => {
                chat = true;
                i += 1
            }
            "--n" => {
                n_gen = Some(args[i + 1].parse().unwrap());
                i += 2
            }
            "--force" => {
                force = Some(args[i + 1].clone());
                i += 2
            }
            "--tokens-out" => {
                tokens_out = Some(args[i + 1].clone());
                i += 2
            }
            other => {
                eprintln!("unknown arg: {}", other);
                std::process::exit(2)
            }
        }
    }
    let mut nt = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1).min(8);
    if let Ok(v) = std::env::var("SMOL_NT") {
        nt = v.parse().unwrap_or(nt);
    }
    let pool = Pool::new(nt);
    let model = load_model(&model_dir, &pool);

    if mode == "ppl" {
        let tin = tokens_in.expect("--tokens required");
        let raw = std::fs::read(&tin).expect("tokens file");
        let ids: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let n = ids.len();
        let ctx_cap = ctx.max(1);
        let mut nc = Caches::new(ctx_cap);
        let msz = CB.min(ctx_cap);
        let mut b = Bufs::new(msz.max(1));
        let mut out = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&logits_out.expect("--logits required")).unwrap());
        let mut logits = vec![0f32; 64 * VV];
        let mut start = 0;
        while start < n {
            let wlen = std::cmp::min(ctx_cap, n - start);
            nc.len = 0;
            let mut off = 0;
            while off < wlen {
                let m = std::cmp::min(msz, wlen - off);
                let chunk = &ids[start + off..start + off + m];
                embed_rows(&model, chunk, &mut b.h);
                forward_block(&model, &pool, &mut nc, &mut b, m);
                // logits for all rows of the chunk, written in blocks of 64
                let mut r0 = 0;
                while r0 < m {
                    let rm = std::cmp::min(64, m - r0);
                    quant_act(&b.h[r0 * H..], rm, H, &mut b.xq, &mut b.xs);
                    matmul(&pool, &b.xq, &b.xs, rm, &model.embed, &mut logits);
                    let bytes = unsafe {
                        std::slice::from_raw_parts(logits.as_ptr() as *const u8, rm * VV * 4)
                    };
                    out.write_all(bytes).unwrap();
                    r0 += rm;
                }
                off += m;
            }
            start += wlen;
        }
        out.flush().unwrap();
        return;
    }

    // ---- gen
    let pf = prompt_file.expect("--prompt-file required");
    let text = std::fs::read_to_string(&pf).expect("prompt file");
    let tok = tokenizers::Tokenizer::from_file(format!("{}/tokenizer.json", model_dir))
        .expect("tokenizer");
    let full = if chat {
        format!("<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", text)
    } else {
        text
    };
    let enc = tok.encode(full.as_str(), false).expect("encode");
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    let forced: Option<Vec<u32>> = force.map(|f| {
        std::fs::read(&f)
            .expect("force file")
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    });
    let ngen = match &forced {
        Some(f) => f.len(),
        None => n_gen.expect("--n or --force required"),
    };
    let plen = prompt_ids.len();
    let ctx_cap = plen + ngen + 1;
    let msz = CB.min(ctx_cap).max(1);
    let mut nc = Caches::new(ctx_cap);
    let mut b = Bufs::new(msz);
    let mut logits = vec![0f32; VV];
    let mut lf = logits_out.map(|p| std::fs::File::create(p).unwrap());
    let mut generated: Vec<u32> = Vec::with_capacity(ngen);
    // prefill
    let mut off = 0;
    while off < plen {
        let m = std::cmp::min(msz, plen - off);
        embed_rows(&model, &prompt_ids[off..off + m], &mut b.h);
        forward_block(&model, &pool, &mut nc, &mut b, m);
        off += m;
    }
    // b.h holds the final chunk starting at chunk_start
    let chunk_start = ((plen.saturating_sub(1)) / msz) * msz;
    let lastrow = plen - 1 - chunk_start;
    quant_act(&b.h[lastrow * H..], 1, H, &mut b.xq, &mut b.xs);
    matmul(&pool, &b.xq, &b.xs, 1, &model.embed, &mut logits);
    let mut cur = argmax(&logits);
    if let Some(f) = &mut lf {
        write_f32s(f, &logits);
    }
    let mut j = 0;
    while j < ngen {
        let t = match &forced {
            Some(f) => f[j],
            None => cur,
        };
        generated.push(t);
        if j + 1 < ngen {
            // decode step with token t
            let ids = [t];
            embed_rows(&model, &ids, &mut b.h);
            forward_block(&model, &pool, &mut nc, &mut b, 1);
            quant_act(&b.h, 1, H, &mut b.xq, &mut b.xs);
            matmul(&pool, &b.xq, &b.xs, 1, &model.embed, &mut logits);
            cur = argmax(&logits);
            if let Some(f) = &mut lf {
                write_f32s(f, &logits);
            }
        }
        j += 1;
    }
    if let Some(f) = &mut lf {
        f.flush().unwrap();
    }
    if let Some(to) = tokens_out {
        let mut f = std::fs::File::create(to).unwrap();
        let mut all = prompt_ids.clone();
        all.extend_from_slice(&generated);
        write_u32s(&mut f, &all);
    }
    dump_stats();
    let txt = tok.decode(&generated, false).unwrap_or_default();
    print!("{}", txt);
    if !txt.ends_with('\n') {
        println!();
    }
}