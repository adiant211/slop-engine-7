// SmolLM2-135M-Instruct CPU inference engine.
//
// Weights: int8 with one scale per output row, stored per worker thread in the order the model reads them, on
// huge pages, so decode is one sequential stream per thread. Activations: int16 with one scale per vector,
// sized so a whole row's dot product fits an i32, so every matrix product is exact integer arithmetic
// (`vpmaddwd`) followed by one float multiply. Attention runs in f32 on a K cache stored 8 positions per
// block (64 x 8 floats) and a row-major V cache, one key/value head per thread.
use std::arch::x86_64::*;
use std::io::Write;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

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
const MAXT: usize = 8; // most worker threads
const CB: usize = 256; // prefill chunk (tokens per forward pass)
const AR: usize = 16; // attention: query rows per block
const PAD: usize = 16384; // slack after each weight arena (prefetch runs past the end)

// Largest activation magnitude for a row of k values: k * 127 * qmax must fit an i32.
const fn qmax(k: usize) -> f32 {
    let q = (i32::MAX as usize) / (127 * k);
    (if q > 32767 { 32767 } else { q }) as f32
}
const QM_H: f32 = qmax(H);
const QM_FF: f32 = qmax(FF);

// ---------------------------------------------------------------- timing (SMOL_TIME=1)
static mut TIMING: bool = false;
static STATS: [AtomicU64; 8] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];
#[inline(always)]
fn tick() -> u64 {
    unsafe {
        if TIMING {
            _rdtsc()
        } else {
            0
        }
    }
}
#[inline(always)]
fn tock(slot: usize, t: &mut u64) {
    unsafe {
        if TIMING {
            let n = _rdtsc();
            STATS[slot].fetch_add(n - *t, Ordering::Relaxed);
            *t = n;
        }
    }
}
fn dump_stats() {
    if unsafe { TIMING } {
        let names = ["norm", "qkv", "attn", "oproj", "gu", "down", "head", "other"];
        let mut s = String::new();
        let tot: u64 = STATS.iter().map(|a| a.load(Ordering::Relaxed)).sum();
        for i in 0..8 {
            let v = STATS[i].load(Ordering::Relaxed);
            s += &format!("{}={:.1}% ", names[i], v as f64 * 100.0 / tot.max(1) as f64);
        }
        eprintln!("STATS total {:.0} Mcyc: {}", tot as f64 / 1e6, s);
    }
}

// ---------------------------------------------------------------- pointers, memory
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

extern "C" {
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
}

// 2 MB-aligned block that asks for transparent huge pages; lives for the whole process.
fn huge_alloc(bytes: usize, zero: bool) -> *mut u8 {
    const A: usize = 2 << 20;
    let len = (bytes.max(1) + A - 1) & !(A - 1);
    unsafe {
        let p = std::alloc::alloc(std::alloc::Layout::from_size_align(len, A).unwrap());
        if p.is_null() {
            eprintln!("out of memory");
            std::process::exit(1);
        }
        madvise(p, len, 14); // MADV_HUGEPAGE
        if zero {
            std::ptr::write_bytes(p, 0, len);
        }
        p
    }
}

// ---------------------------------------------------------------- thread pool
struct St {
    slot: std::cell::UnsafeCell<[usize; 2]>,
    epoch: AtomicU64,
    done: AtomicUsize,
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
            done: AtomicUsize::new(0),
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
    let base = total / n;
    let rem = total % n;
    let lo = tid * base + tid.min(rem);
    let hi = lo + base + if tid < rem { 1 } else { 0 };
    (lo, hi)
}

// Run f(row) for every row; rows are shared between the threads when there are enough of them.
#[inline]
fn par_rows<F: Fn(usize) + Sync>(pool: &Pool, m: usize, f: F) {
    if m < 8 || pool.n == 1 {
        for r in 0..m {
            f(r);
        }
    } else {
        let nt = pool.n;
        pool.run(|tid| {
            let (lo, hi) = split(m, nt, tid);
            for r in lo..hi {
                f(r);
            }
        });
    }
}

// ---------------------------------------------------------------- small vector helpers
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
unsafe fn hmax(a: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(a);
    let hi = _mm256_extractf128_ps::<1>(a);
    let s = _mm_max_ps(lo, hi);
    let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_max_ss(s, _mm_shuffle_ps(s, s, 1));
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
static MASKS: [i32; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0];
// first n of 8 lanes set
#[inline(always)]
unsafe fn mask8(n: usize) -> __m256 {
    _mm256_loadu_ps(MASKS.as_ptr().add(8 - n) as *const f32)
}

#[inline(always)]
unsafe fn exp256(x: __m256) -> __m256 {
    let x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0)), _mm256_set1_ps(87.0));
    let n = _mm256_round_ps(
        _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693145751953125), x);
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(1.428606765330187e-6), r);
    let mut p = _mm256_set1_ps(1.0 / 5040.0);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 720.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 120.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 24.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 6.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
    let ex = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    _mm256_mul_ps(p, _mm256_castsi256_ps(ex))
}

// ---------------------------------------------------------------- quantisation
// 8 bf16 values -> f32
#[inline(always)]
unsafe fn bf16x8(p: *const u16) -> __m256 {
    _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(p as *const __m128i)), 16))
}

// One weight row (k bf16 values, k % 32 == 0) -> int8 in [-127, 127]; returns the row's scale.
unsafe fn quant_w_row(src: *const u16, k: usize, dst: *mut u8) -> f32 {
    let absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let mut mx = _mm256_setzero_ps();
    let mut i = 0;
    while i < k {
        mx = _mm256_max_ps(mx, _mm256_and_ps(bf16x8(src.add(i)), absm));
        i += 8;
    }
    let m = hmax(mx);
    let s = if m > 0.0 { m / 127.0 } else { 1.0 };
    let inv = _mm256_set1_ps(1.0 / s);
    let lim = _mm256_set1_ps(127.0);
    let nlim = _mm256_set1_ps(-127.0);
    let idx = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let mut i = 0;
    while i < k {
        let mut v = [_mm256_setzero_si256(); 4];
        for t in 0..4 {
            let f = _mm256_mul_ps(bf16x8(src.add(i + t * 8)), inv);
            v[t] = _mm256_cvtps_epi32(_mm256_max_ps(_mm256_min_ps(f, lim), nlim));
        }
        let b = _mm256_packs_epi16(_mm256_packs_epi32(v[0], v[1]), _mm256_packs_epi32(v[2], v[3]));
        _mm256_storeu_si256(dst.add(i) as *mut __m256i, _mm256_permutevar8x32_epi32(b, idx));
        i += 32;
    }
    s
}

// One activation row (k f32, k % 16 == 0) -> int16 in [-qm, qm]; returns the scale (0 for an all-zero row).
#[inline(always)]
unsafe fn quant_row(x: *const f32, k: usize, q: *mut i16, qm: f32) -> f32 {
    let absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let mut mx = _mm256_setzero_ps();
    let mut i = 0;
    while i < k {
        mx = _mm256_max_ps(mx, _mm256_and_ps(_mm256_loadu_ps(x.add(i)), absm));
        i += 8;
    }
    let m = hmax(mx);
    if !(m > 0.0) || !m.is_finite() {
        std::ptr::write_bytes(q, 0, k);
        return 0.0;
    }
    let inv = _mm256_set1_ps(qm / m);
    let mut i = 0;
    while i < k {
        let a = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x.add(i)), inv));
        let b = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x.add(i + 8)), inv));
        let p = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packs_epi32(a, b));
        _mm256_storeu_si256(q.add(i) as *mut __m256i, p);
        i += 16;
    }
    m / qm
}

// h (H f32) -> RMSNorm with weight w -> int16 row; returns its scale.
#[inline(always)]
unsafe fn rmsnorm_quant(h: *const f32, w: *const f32, q: *mut i16) -> f32 {
    let mut a = _mm256_setzero_ps();
    let mut i = 0;
    while i < H {
        let v = _mm256_loadu_ps(h.add(i));
        a = _mm256_fmadd_ps(v, v, a);
        i += 8;
    }
    let scale = 1.0 / ((hsum(a) / (H as f32)) + EPS).sqrt();
    let sv = _mm256_set1_ps(scale);
    let mut x = [0f32; H];
    let mut i = 0;
    while i < H {
        let v = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(h.add(i)), sv), _mm256_loadu_ps(w.add(i)));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), v);
        i += 8;
    }
    quant_row(x.as_ptr(), H, q, QM_H)
}

#[inline(always)]
unsafe fn add_row(h: *mut f32, t: *const f32) {
    let mut i = 0;
    while i < H {
        _mm256_storeu_ps(h.add(i), _mm256_add_ps(_mm256_loadu_ps(h.add(i)), _mm256_loadu_ps(t.add(i))));
        i += 8;
    }
}

// gu row = [gate (FF) | up (FF)]: gate <- silu(gate) * up, then quantise gate.
#[inline(always)]
unsafe fn silu_quant(gu: *mut f32, q: *mut i16) -> f32 {
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    while i < FF {
        let x = _mm256_loadu_ps(gu.add(i));
        let e = exp256(_mm256_sub_ps(zero, x));
        let s = _mm256_div_ps(x, _mm256_add_ps(one, e));
        _mm256_storeu_ps(gu.add(i), _mm256_mul_ps(s, _mm256_loadu_ps(gu.add(FF + i))));
        i += 8;
    }
    quant_row(gu, FF, q, QM_FF)
}

// ---------------------------------------------------------------- integer matrix kernels
// Weight matrix n x k: thread t holds rows lo[t]..lo[t+1] at part[t] (int8, row-major); sc = per-row scale.
struct Mat {
    n: usize,
    k: usize,
    sc: Vec<f32>,
    part: [P<u8>; MAXT],
    lo: [usize; MAXT + 1],
}

static mut PF_DIST: usize = 4096;

#[inline(always)]
unsafe fn dot_row(w: *const u8, x: *const i16, k: usize) -> i32 {
    let mut a0 = _mm256_setzero_si256();
    let mut a1 = _mm256_setzero_si256();
    let mut i = 0;
    while i < k {
        let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.add(i) as *const __m128i));
        let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.add(i + 16) as *const __m128i));
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, _mm256_loadu_si256(x.add(i) as *const __m256i)));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(w1, _mm256_loadu_si256(x.add(i + 16) as *const __m256i)));
        i += 32;
    }
    hsum_i32(_mm256_add_epi32(a0, a1))
}

// rows x one activation vector; streams the weights once. Returns (best value, best index) when track is set.
#[inline(always)]
unsafe fn gemv_k<const K: usize>(
    w: *const u8,
    rows: usize,
    x: *const i16,
    sc: *const f32,
    xs: f32,
    out: *mut f32,
    track: bool,
) -> (f32, usize) {
    let pf = PF_DIST;
    let mut bv = f32::NEG_INFINITY;
    let mut bi = 0usize;
    for j in 0..rows {
        let wr = w.add(j * K);
        if pf != 0 {
            let mut c = 0;
            while c < K {
                _mm_prefetch::<_MM_HINT_T0>(wr.wrapping_add(pf + c) as *const i8);
                c += 64;
            }
        }
        let v = dot_row(wr, x, K) as f32 * (*sc.add(j) * xs);
        *out.add(j) = v;
        if track && v > bv {
            bv = v;
            bi = j;
        }
    }
    (bv, bi)
}

unsafe fn gemv(w: *const u8, k: usize, rows: usize, x: *const i16, sc: *const f32, xs: f32, out: *mut f32, track: bool) -> (f32, usize) {
    if k == H {
        gemv_k::<H>(w, rows, x, sc, xs, out, track)
    } else {
        gemv_k::<FF>(w, rows, x, sc, xs, out, track)
    }
}

// NW weight rows x NX activation rows, full dot products in i32.
#[inline(always)]
unsafe fn tile<const NW: usize, const NX: usize>(
    w: *const u8,
    k: usize,
    x: *const i16,
    sc: *const f32,
    xs: *const f32,
    out: *mut f32,
    ostride: usize,
) {
    let mut acc = [[_mm256_setzero_si256(); NX]; NW];
    let mut t = 0;
    while t < k {
        let mut wv = [_mm256_setzero_si256(); NW];
        for a in 0..NW {
            wv[a] = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.add(a * k + t) as *const __m128i));
        }
        for r in 0..NX {
            let xv = _mm256_loadu_si256(x.add(r * k + t) as *const __m256i);
            for a in 0..NW {
                acc[a][r] = _mm256_add_epi32(acc[a][r], _mm256_madd_epi16(wv[a], xv));
            }
        }
        t += 16;
    }
    for a in 0..NW {
        let s = *sc.add(a);
        for r in 0..NX {
            *out.add(r * ostride + a) = hsum_i32(acc[a][r]) as f32 * (s * *xs.add(r));
        }
    }
}

// rows of W (this thread's share) x m activation rows, blocked so the activations stay in L1 and the
// weights in L2. out points at column 0 of this share; token rows are ostride apart.
unsafe fn gemm_part(
    w: *const u8,
    k: usize,
    rows: usize,
    x: *const i16,
    xs: *const f32,
    m: usize,
    sc: *const f32,
    out: *mut f32,
    ostride: usize,
) {
    let wch = ((192 * 1024 / k) & !1).max(2);
    let xb = (((20 * 1024) / (2 * k)) & !3).clamp(4, 32);
    let mut j0 = 0;
    while j0 < rows {
        let j1 = (j0 + wch).min(rows);
        let mut i0 = 0;
        while i0 < m {
            let i1 = (i0 + xb).min(m);
            let mut j = j0;
            while j < j1 {
                let nw = (j1 - j).min(2);
                let wp = w.add(j * k);
                let mut i = i0;
                while i < i1 {
                    let nx = (i1 - i).min(4);
                    let xp = x.add(i * k);
                    let op = out.add(i * ostride + j);
                    let (scp, xsp) = (sc.add(j), xs.add(i));
                    match (nw, nx) {
                        (2, 4) => tile::<2, 4>(wp, k, xp, scp, xsp, op, ostride),
                        (2, 3) => tile::<2, 3>(wp, k, xp, scp, xsp, op, ostride),
                        (2, 2) => tile::<2, 2>(wp, k, xp, scp, xsp, op, ostride),
                        (2, _) => tile::<2, 1>(wp, k, xp, scp, xsp, op, ostride),
                        (_, 4) => tile::<1, 4>(wp, k, xp, scp, xsp, op, ostride),
                        (_, 3) => tile::<1, 3>(wp, k, xp, scp, xsp, op, ostride),
                        (_, 2) => tile::<1, 2>(wp, k, xp, scp, xsp, op, ostride),
                        _ => tile::<1, 1>(wp, k, xp, scp, xsp, op, ostride),
                    }
                    i += nx;
                }
                j += nw;
            }
            i0 = i1;
        }
        j0 = j1;
    }
}

// out[i * ostride + j] = sum_c x[i][c] * W[j][c], for m activation rows.
fn matmul(pool: &Pool, w: &Mat, xq: *const i16, xs: *const f32, m: usize, out: *mut f32, ostride: usize) {
    let (px, ps, po, pw) = (P(xq), P(xs), PM(out), P(w as *const Mat));
    pool.run(move |tid| unsafe {
        let w = &*pw.get();
        let (lo, hi) = (w.lo[tid], w.lo[tid + 1]);
        if lo >= hi {
            return;
        }
        let wp = w.part[tid].get();
        let sc = w.sc.as_ptr().add(lo);
        if m == 1 {
            gemv(wp, w.k, hi - lo, px.get(), sc, *ps.get(), po.get().add(lo), false);
        } else {
            gemm_part(wp, w.k, hi - lo, px.get(), ps.get(), m, sc, po.get().add(lo), ostride);
        }
    });
}

// One activation row x the output head: fills logits, returns the argmax (lowest id on a tie).
fn head_argmax(pool: &Pool, w: &Mat, xq: *const i16, xs: f32, logits: *mut f32) -> u32 {
    let mut best = [(f32::NEG_INFINITY, 0usize); MAXT];
    let (px, po, pw, pb) = (P(xq), PM(logits), P(w as *const Mat), PM(best.as_mut_ptr()));
    pool.run(move |tid| unsafe {
        let w = &*pw.get();
        let (lo, hi) = (w.lo[tid], w.lo[tid + 1]);
        if lo >= hi {
            return;
        }
        let (v, i) = gemv(w.part[tid].get(), w.k, hi - lo, px.get(), w.sc.as_ptr().add(lo), xs, po.get().add(lo), true);
        *pb.get().add(tid) = (v, lo + i);
    });
    let mut bv = f32::NEG_INFINITY;
    let mut bi = 0usize;
    for t in 0..pool.n {
        if best[t].0 > bv {
            bv = best[t].0;
            bi = best[t].1;
        }
    }
    bi as u32
}

// ---------------------------------------------------------------- model
struct Layer {
    wqkv: Mat,
    wo: Mat,
    wgu: Mat,
    wdown: Mat,
    an: Vec<f32>,
    fnn: Vec<f32>,
}

struct Model {
    layers: Vec<Layer>,
    head: Mat,
    norm: Vec<f32>,
    emb: P<u16>, // bf16 [VV][H], the file's own bytes
}

struct Src {
    parts: Vec<(P<u8>, usize)>, // (bf16 data, rows)
    n: usize,
    k: usize,
}
impl Src {
    #[inline]
    fn row(&self, mut r: usize) -> *const u16 {
        for (p, n) in &self.parts {
            if r < *n {
                return unsafe { p.get().add(r * self.k * 2) as *const u16 };
            }
            r -= *n;
        }
        unreachable!()
    }
}

fn load_model(dir: &str, pool: &Pool) -> Model {
    let path = format!("{}/model.safetensors", dir);
    let data: &'static [u8] = Vec::leak(std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {}", path, e);
        std::process::exit(1)
    }));
    let st = safetensors::SafeTensors::deserialize(data).expect("bad safetensors");
    let get = |name: &str, n: usize, k: usize| -> (P<u8>, usize) {
        let t = st.tensor(name).unwrap_or_else(|e| panic!("tensor {} {:?}", name, e));
        let sh = t.shape();
        if k == 0 {
            assert_eq!(sh[0], n, "{} shape", name);
        } else {
            assert_eq!((sh[0], sh[1]), (n, k), "{} shape", name);
        }
        (P(t.data().as_ptr()), n)
    };
    let vecf = |name: &str| -> Vec<f32> {
        let (p, n) = get(name, H, 0);
        (0..n).map(|i| f32::from_bits((unsafe { (p.get().add(i * 2) as *const u16).read_unaligned() } as u32) << 16)).collect()
    };
    // matrices in the order a token reads them
    let mut srcs: Vec<Src> = Vec::with_capacity(NL * 4 + 1);
    let mut norms: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(NL);
    for l in 0..NL {
        let p = format!("model.layers.{}", l);
        srcs.push(Src {
            parts: vec![
                get(&format!("{}.self_attn.q_proj.weight", p), QDIM, H),
                get(&format!("{}.self_attn.k_proj.weight", p), KVDIM, H),
                get(&format!("{}.self_attn.v_proj.weight", p), KVDIM, H),
            ],
            n: QKV,
            k: H,
        });
        srcs.push(Src { parts: vec![get(&format!("{}.self_attn.o_proj.weight", p), H, QDIM)], n: H, k: QDIM });
        srcs.push(Src {
            parts: vec![
                get(&format!("{}.mlp.gate_proj.weight", p), FF, H),
                get(&format!("{}.mlp.up_proj.weight", p), FF, H),
            ],
            n: GU,
            k: H,
        });
        srcs.push(Src { parts: vec![get(&format!("{}.mlp.down_proj.weight", p), H, FF)], n: H, k: FF });
        norms.push((
            vecf(&format!("{}.input_layernorm.weight", p)),
            vecf(&format!("{}.post_attention_layernorm.weight", p)),
        ));
    }
    let emb = get("model.embed_tokens.weight", VV, H);
    srcs.push(Src { parts: vec![emb], n: VV, k: H });
    let norm = vecf("model.norm.weight");

    let nt = pool.n;
    let mut mats: Vec<Mat> = Vec::with_capacity(srcs.len());
    let mut size = [0usize; MAXT];
    for s in &srcs {
        let mut mt = Mat { n: s.n, k: s.k, sc: vec![0f32; s.n], part: [P(std::ptr::null()); MAXT], lo: [s.n; MAXT + 1] };
        for t in 0..nt {
            let (lo, hi) = split(s.n, nt, t);
            mt.lo[t] = lo;
            mt.lo[t + 1] = hi;
            mt.part[t] = P(size[t] as *const u8); // offset for now
            size[t] += (hi - lo) * s.k;
        }
        mats.push(mt);
    }
    let mut arena = [PM(std::ptr::null_mut::<u8>()); MAXT];
    for t in 0..nt {
        arena[t] = PM(huge_alloc(size[t] + PAD, false));
        for mt in mats.iter_mut() {
            mt.part[t] = P(unsafe { arena[t].get().add(mt.part[t].get() as usize) } as *const u8);
        }
    }
    {
        let scs: Vec<PM<f32>> = mats.iter_mut().map(|m| PM(m.sc.as_mut_ptr())).collect();
        let (psrc, pmat, psc) = (P(srcs.as_ptr()), P(mats.as_ptr()), P(scs.as_ptr()));
        let nm = srcs.len();
        pool.run(move |tid| unsafe {
            let mut dst = arena[tid].get();
            for mi in 0..nm {
                let s = &*psrc.get().add(mi);
                let mt = &*pmat.get().add(mi);
                let sc = (*psc.get().add(mi)).get();
                for r in mt.lo[tid]..mt.lo[tid + 1] {
                    *sc.add(r) = quant_w_row(s.row(r), s.k, dst);
                    dst = dst.add(s.k);
                }
            }
            std::ptr::write_bytes(arena[tid].get().add(size[tid]), 0, PAD);
        });
    }
    let head = mats.pop().unwrap();
    let mut it = mats.into_iter();
    let mut layers = Vec::with_capacity(NL);
    for (an, fnn) in norms {
        let wqkv = it.next().unwrap();
        let wo = it.next().unwrap();
        let wgu = it.next().unwrap();
        let wdown = it.next().unwrap();
        layers.push(Layer { wqkv, wo, wgu, wdown, an, fnn });
    }
    Model { layers, head, norm, emb: P(emb.0.get() as *const u16) }
}

// ---------------------------------------------------------------- state
// K: [kv head][layer][block of 8 positions][64 dims][8]   V: [kv head][layer][position][64]
struct Caches {
    ctx: usize,
    nb: usize,
    k: PM<f32>,
    v: PM<f32>,
    cos: Vec<f32>, // [ctx][32]
    sin: Vec<f32>,
    len: usize,
}
impl Caches {
    fn new(ctx: usize) -> Caches {
        let nb = (ctx + 7) / 8;
        let mut cos = vec![0f32; ctx * 32];
        let mut sin = vec![0f32; ctx * 32];
        for i in 0..32 {
            let inv = (1.0f64 / (100000.0f64).powf((2 * i) as f64 / 64.0)) as f32;
            for p in 0..ctx {
                let a = p as f32 * inv;
                cos[p * 32 + i] = a.cos();
                sin[p * 32 + i] = a.sin();
            }
        }
        Caches {
            ctx,
            nb,
            k: PM(huge_alloc(NL * NKV * nb * 512 * 4, true) as *mut f32),
            v: PM(huge_alloc(NL * NKV * ctx * 64 * 4, true) as *mut f32),
            cos,
            sin,
            len: 0,
        }
    }
}

struct Bufs {
    h: Vec<f32>,
    tmp: Vec<f32>,
    qkv: Vec<f32>,
    ao: Vec<f32>,
    gu: Vec<f32>,
    xq: Vec<i16>,
    xs: Vec<f32>,
    aq: Vec<i16>,
    az: Vec<f32>,
    gq: Vec<i16>,
    gs: Vec<f32>,
    scr: Vec<PM<f32>>, // attention scores, one scratch per thread
}
impl Bufs {
    fn new(m: usize, ctx: usize, nt: usize) -> Bufs {
        let rows = AR.min(m);
        Bufs {
            h: vec![0f32; m * H],
            tmp: vec![0f32; m * H],
            qkv: vec![0f32; m * QKV],
            ao: vec![0f32; m * QDIM],
            gu: vec![0f32; m * GU],
            xq: vec![0i16; m * H],
            xs: vec![0f32; m],
            aq: vec![0i16; m * QDIM],
            az: vec![0f32; m],
            gq: vec![0i16; m * FF],
            gs: vec![0f32; m],
            scr: (0..nt).map(|_| PM(Vec::leak(vec![0f32; rows * 3 * (ctx + 16)]).as_mut_ptr())).collect(),
        }
    }
}

// ---------------------------------------------------------------- attention
#[inline(always)]
unsafe fn rope64(v: *mut f32, cs: *const f32, sn: *const f32) {
    let mut d = 0;
    while d < 32 {
        let a = _mm256_loadu_ps(v.add(d));
        let b = _mm256_loadu_ps(v.add(d + 32));
        let c = _mm256_loadu_ps(cs.add(d));
        let s = _mm256_loadu_ps(sn.add(d));
        _mm256_storeu_ps(v.add(d), _mm256_fnmadd_ps(b, s, _mm256_mul_ps(a, c)));
        _mm256_storeu_ps(v.add(d + 32), _mm256_fmadd_ps(a, s, _mm256_mul_ps(b, c)));
        d += 8;
    }
}

// 3 query heads (q: 3 x 64) against C key blocks (8 positions each); writes 8 * C scaled scores per head.
#[inline(always)]
unsafe fn score_tile<const C: usize>(q: *const f32, kb: *const f32, out: *mut f32, sstride: usize) {
    let mut acc = [[_mm256_setzero_ps(); C]; 3];
    for d in 0..64 {
        let q0 = _mm256_broadcast_ss(&*q.add(d));
        let q1 = _mm256_broadcast_ss(&*q.add(64 + d));
        let q2 = _mm256_broadcast_ss(&*q.add(128 + d));
        for c in 0..C {
            let kv = _mm256_loadu_ps(kb.add(c * 512 + d * 8));
            acc[0][c] = _mm256_fmadd_ps(q0, kv, acc[0][c]);
            acc[1][c] = _mm256_fmadd_ps(q1, kv, acc[1][c]);
            acc[2][c] = _mm256_fmadd_ps(q2, kv, acc[2][c]);
        }
    }
    let s = _mm256_set1_ps(0.125);
    for x in 0..3 {
        for c in 0..C {
            _mm256_storeu_ps(out.add(x * sstride + c * 8), _mm256_mul_ps(acc[x][c], s));
        }
    }
}

// scores s[0..len] -> probabilities, in place. The buffer holds whole blocks of 8.
#[inline(always)]
unsafe fn softmax(s: *mut f32, len: usize) {
    let full = len & !7;
    let rem = len & 7;
    let ninf = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut mx = ninf;
    let mut i = 0;
    while i < full {
        mx = _mm256_max_ps(mx, _mm256_loadu_ps(s.add(i)));
        i += 8;
    }
    let mk = mask8(rem);
    if rem != 0 {
        mx = _mm256_max_ps(mx, _mm256_blendv_ps(ninf, _mm256_loadu_ps(s.add(full)), mk));
    }
    let m = _mm256_set1_ps(hmax(mx));
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i < full {
        let e = exp256(_mm256_sub_ps(_mm256_loadu_ps(s.add(i)), m));
        _mm256_storeu_ps(s.add(i), e);
        sum = _mm256_add_ps(sum, e);
        i += 8;
    }
    if rem != 0 {
        let e = _mm256_and_ps(exp256(_mm256_sub_ps(_mm256_blendv_ps(m, _mm256_loadu_ps(s.add(full)), mk), m)), mk);
        _mm256_storeu_ps(s.add(full), e);
        sum = _mm256_add_ps(sum, e);
    }
    let inv = _mm256_set1_ps(1.0 / hsum(sum));
    let mut i = 0;
    while i < len {
        _mm256_storeu_ps(s.add(i), _mm256_mul_ps(_mm256_loadu_ps(s.add(i)), inv));
        i += 8;
    }
}

// out[0..64] (+)= sum_j p[j] * v[j][0..64]
#[inline(always)]
unsafe fn pv_row(p: *const f32, v: *const f32, n: usize, out: *mut f32, first: bool) {
    let mut acc = [_mm256_setzero_ps(); 8];
    if !first {
        for t in 0..8 {
            acc[t] = _mm256_loadu_ps(out.add(t * 8));
        }
    }
    for j in 0..n {
        let pj = _mm256_broadcast_ss(&*p.add(j));
        let vr = v.add(j * 64);
        for t in 0..8 {
            acc[t] = _mm256_fmadd_ps(pj, _mm256_loadu_ps(vr.add(t * 8)), acc[t]);
        }
    }
    for t in 0..8 {
        _mm256_storeu_ps(out.add(t * 8), acc[t]);
    }
}

// One key/value head (its three query heads) for m new rows at positions pos0..: rotate q and k, append
// K and V to the cache, then attend. qkv rows are QKV apart, ao rows QDIM apart.
unsafe fn attn_head(nc: &Caches, l: usize, kh: usize, qkv: *mut f32, ao: *mut f32, m: usize, pos0: usize, scr: *mut f32) {
    let kb = nc.k.get().add((kh * NL + l) * nc.nb * 512);
    let vb = nc.v.get().add((kh * NL + l) * nc.ctx * 64);
    for i in 0..m {
        let p = pos0 + i;
        let (cs, sn) = (nc.cos.as_ptr().add(p * 32), nc.sin.as_ptr().add(p * 32));
        let row = qkv.add(i * QKV);
        for x in 0..3 {
            rope64(row.add((kh * 3 + x) * 64), cs, sn);
        }
        let kr = row.add(QDIM + kh * 64);
        rope64(kr, cs, sn);
        let kd = kb.add((p >> 3) * 512 + (p & 7));
        for d in 0..64 {
            *kd.add(d * 8) = *kr.add(d);
        }
        std::ptr::copy_nonoverlapping(row.add(QDIM + KVDIM + kh * 64) as *const f32, vb.add(p * 64), 64);
    }
    let mut i0 = 0;
    while i0 < m {
        let rows = AR.min(m - i0);
        let maxlen = pos0 + i0 + rows;
        let nbmax = (maxlen + 7) >> 3;
        let ss = nbmax * 8; // scores: [(r * 3 + x)][ss]
        // scores, key blocks in the outer loop so they stay in L1 across the query rows
        let mut g = 0;
        while g < nbmax {
            for r in 0..rows {
                let nbr = (pos0 + i0 + r + 8) >> 3;
                if g >= nbr {
                    continue;
                }
                let q = qkv.add((i0 + r) * QKV + kh * 3 * 64) as *const f32;
                let o = scr.add(r * 3 * ss + g * 8);
                let kp = kb.add(g * 512) as *const f32;
                match (nbr - g).min(4) {
                    4 => score_tile::<4>(q, kp, o, ss),
                    3 => score_tile::<3>(q, kp, o, ss),
                    2 => score_tile::<2>(q, kp, o, ss),
                    _ => score_tile::<1>(q, kp, o, ss),
                }
            }
            g += 4;
        }
        for r in 0..rows {
            for x in 0..3 {
                softmax(scr.add((r * 3 + x) * ss), pos0 + i0 + r + 1);
            }
        }
        // weighted sums of V, 64 positions at a time
        let mut j0 = 0;
        while j0 < maxlen {
            for r in 0..rows {
                let len = pos0 + i0 + r + 1;
                if j0 >= len {
                    continue;
                }
                let n = (len - j0).min(64);
                for x in 0..3 {
                    pv_row(
                        scr.add((r * 3 + x) * ss + j0),
                        vb.add(j0 * 64),
                        n,
                        ao.add((i0 + r) * QDIM + (kh * 3 + x) * 64),
                        j0 == 0,
                    );
                }
            }
            j0 += 64;
        }
        i0 += rows;
    }
}

fn attention(pool: &Pool, nc: &Caches, b: &mut Bufs, l: usize, m: usize, pos0: usize) {
    let nt = pool.n;
    let (pc, pq, pa, ps) = (P(nc as *const Caches), PM(b.qkv.as_mut_ptr()), PM(b.ao.as_mut_ptr()), P(b.scr.as_ptr()));
    pool.run(move |tid| unsafe {
        let scr = (*ps.get().add(tid)).get();
        let mut kh = tid;
        while kh < NKV {
            attn_head(&*pc.get(), l, kh, pq.get(), pa.get(), m, pos0, scr);
            kh += nt;
        }
    });
}

// ---------------------------------------------------------------- forward
// b.h holds m embedded rows; on return b.xq / b.xs hold the final-norm rows, quantised for the head.
fn forward(md: &Model, pool: &Pool, nc: &mut Caches, b: &mut Bufs, m: usize) {
    let pos0 = nc.len;
    assert!(pos0 + m <= nc.ctx);
    let (ph, ptmp) = (PM(b.h.as_mut_ptr()), PM(b.tmp.as_mut_ptr()));
    let (pxq, pxs) = (PM(b.xq.as_mut_ptr()), PM(b.xs.as_mut_ptr()));
    let (pao, paq, paz) = (PM(b.ao.as_mut_ptr()), PM(b.aq.as_mut_ptr()), PM(b.az.as_mut_ptr()));
    let (pgu, pgq, pgs) = (PM(b.gu.as_mut_ptr()), PM(b.gq.as_mut_ptr()), PM(b.gs.as_mut_ptr()));
    let mut t = tick();
    {
        let w = P(md.layers[0].an.as_ptr());
        par_rows(pool, m, move |r| unsafe {
            *pxs.get().add(r) = rmsnorm_quant(ph.get().add(r * H), w.get(), pxq.get().add(r * H));
        });
    }
    tock(0, &mut t);
    for l in 0..NL {
        let lay = &md.layers[l];
        matmul(pool, &lay.wqkv, pxq.get(), pxs.get(), m, b.qkv.as_mut_ptr(), QKV);
        tock(1, &mut t);
        attention(pool, nc, b, l, m, pos0);
        tock(2, &mut t);
        par_rows(pool, m, move |r| unsafe {
            *paz.get().add(r) = quant_row(pao.get().add(r * QDIM), QDIM, paq.get().add(r * QDIM), QM_H);
        });
        matmul(pool, &lay.wo, paq.get(), paz.get(), m, ptmp.get(), H);
        tock(3, &mut t);
        let w = P(lay.fnn.as_ptr());
        par_rows(pool, m, move |r| unsafe {
            add_row(ph.get().add(r * H), ptmp.get().add(r * H));
            *pxs.get().add(r) = rmsnorm_quant(ph.get().add(r * H), w.get(), pxq.get().add(r * H));
        });
        tock(0, &mut t);
        matmul(pool, &lay.wgu, pxq.get(), pxs.get(), m, pgu.get(), GU);
        tock(4, &mut t);
        par_rows(pool, m, move |r| unsafe {
            *pgs.get().add(r) = silu_quant(pgu.get().add(r * GU), pgq.get().add(r * FF));
        });
        tock(0, &mut t);
        matmul(pool, &lay.wdown, pgq.get(), pgs.get(), m, ptmp.get(), H);
        tock(5, &mut t);
        let w = P(if l + 1 < NL { md.layers[l + 1].an.as_ptr() } else { md.norm.as_ptr() });
        par_rows(pool, m, move |r| unsafe {
            add_row(ph.get().add(r * H), ptmp.get().add(r * H));
            *pxs.get().add(r) = rmsnorm_quant(ph.get().add(r * H), w.get(), pxq.get().add(r * H));
        });
        tock(0, &mut t);
    }
    nc.len += m;
}

fn embed_rows(md: &Model, ids: &[u32], h: &mut [f32]) {
    for (i, &id) in ids.iter().enumerate() {
        let r = (id as usize).min(VV - 1);
        unsafe {
            let src = md.emb.get().add(r * H);
            let dst = h.as_mut_ptr().add(i * H);
            let mut j = 0;
            while j < H {
                _mm256_storeu_ps(dst.add(j), bf16x8(src.add(j)));
                j += 8;
            }
        }
    }
}

fn read_u32s(path: &str) -> Vec<u32> {
    let raw = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {}", path, e);
        std::process::exit(1)
    });
    raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
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
    let val = |i: usize| -> String {
        args.get(i + 1).cloned().unwrap_or_else(|| {
            eprintln!("missing value for {}", args[i]);
            std::process::exit(2)
        })
    };
    let num = |i: usize| -> usize {
        val(i).parse().unwrap_or_else(|_| {
            eprintln!("bad number for {}", args[i]);
            std::process::exit(2)
        })
    };
    while i < args.len() {
        match args[i].as_str() {
            "--model" => model_dir = val(i),
            "--tokens" => tokens_in = Some(val(i)),
            "--ctx" => ctx = num(i),
            "--logits" => logits_out = Some(val(i)),
            "--prompt-file" => prompt_file = Some(val(i)),
            "--n" => n_gen = Some(num(i)),
            "--force" => force = Some(val(i)),
            "--tokens-out" => tokens_out = Some(val(i)),
            "--chat" => {
                chat = true;
                i += 1;
                continue;
            }
            other => {
                eprintln!("unknown arg: {}", other);
                std::process::exit(2)
            }
        }
        i += 2;
    }
    unsafe {
        TIMING = std::env::var("SMOL_TIME").is_ok();
        if let Ok(v) = std::env::var("SMOL_PF") {
            PF_DIST = v.parse().unwrap_or(PF_DIST);
        }
    }
    let mut nt = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1);
    if let Ok(v) = std::env::var("SMOL_NT") {
        nt = v.parse().unwrap_or(nt);
    }
    let nt = nt.clamp(1, MAXT);
    let pool = Pool::new(nt);
    let model = load_model(&model_dir, &pool);

    if mode == "ppl" {
        let ids = read_u32s(&tokens_in.expect("--tokens required"));
        let n = ids.len();
        let win = ctx.max(1).min(n.max(1));
        let mut nc = Caches::new(win);
        let msz = CB.min(win);
        let mut b = Bufs::new(msz, win, nt);
        let mut out = std::fs::File::create(&logits_out.expect("--logits required")).unwrap();
        let lrows = 32.min(msz);
        let mut logits = vec![0f32; lrows * VV];
        let mut start = 0;
        while start < n {
            let wlen = win.min(n - start);
            nc.len = 0;
            let mut off = 0;
            while off < wlen {
                let m = msz.min(wlen - off);
                embed_rows(&model, &ids[start + off..start + off + m], &mut b.h);
                forward(&model, &pool, &mut nc, &mut b, m);
                let mut t = tick();
                let mut r0 = 0;
                while r0 < m {
                    let rm = lrows.min(m - r0);
                    matmul(&pool, &model.head, b.xq[r0 * H..].as_ptr(), b.xs[r0..].as_ptr(), rm, logits.as_mut_ptr(), VV);
                    out.write_all(as_bytes(&logits[..rm * VV])).unwrap();
                    r0 += rm;
                }
                tock(6, &mut t);
                off += m;
            }
            start += wlen;
        }
        out.flush().unwrap();
        dump_stats();
        return;
    }
    if mode != "gen" {
        eprintln!("unknown mode: {}", mode);
        std::process::exit(2);
    }

    // ---- gen
    let pf = prompt_file.expect("--prompt-file required");
    let text = String::from_utf8_lossy(&std::fs::read(&pf).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {}", pf, e);
        std::process::exit(1)
    }))
    .into_owned();
    let tok = tokenizers::Tokenizer::from_file(format!("{}/tokenizer.json", model_dir)).expect("tokenizer");
    let full = if chat {
        format!("<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", text)
    } else {
        text
    };
    let enc = tok.encode(full.as_str(), false).expect("encode");
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    let forced: Option<Vec<u32>> = force.map(|f| read_u32s(&f));
    let ngen = match &forced {
        Some(f) => f.len(),
        None => n_gen.expect("--n or --force required"),
    };
    // an empty prompt has nothing to condition on: start from the end-of-text token (not reported)
    let ctx_ids: Vec<u32> = if prompt_ids.is_empty() { vec![0] } else { prompt_ids.clone() };
    let plen = ctx_ids.len();
    let ctx_cap = plen + ngen;
    let msz = CB.min(plen);
    let mut nc = Caches::new(ctx_cap);
    let mut b = Bufs::new(msz, ctx_cap, nt);
    let mut logits = vec![0f32; VV];
    let mut lf = logits_out.map(|p| std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(p).unwrap()));
    let mut generated: Vec<u32> = Vec::with_capacity(ngen);
    let mut off = 0;
    let mut m = 0;
    while off < plen {
        m = msz.min(plen - off);
        embed_rows(&model, &ctx_ids[off..off + m], &mut b.h);
        forward(&model, &pool, &mut nc, &mut b, m);
        off += m;
    }
    let mut t = tick();
    for j in 0..ngen {
        // the row that predicts token j: the prompt's last row, then each fed token's row
        let row = if j == 0 { m - 1 } else { 0 };
        let cur = head_argmax(&pool, &model.head, b.xq[row * H..].as_ptr(), b.xs[row], logits.as_mut_ptr());
        tock(6, &mut t);
        if let Some(f) = &mut lf {
            f.write_all(as_bytes(&logits)).unwrap();
        }
        let tk = match &forced {
            Some(f) => f[j],
            None => cur,
        };
        generated.push(tk);
        if j + 1 < ngen {
            embed_rows(&model, &[tk], &mut b.h);
            t = tick();
            forward(&model, &pool, &mut nc, &mut b, 1);
            t = tick();
        }
    }
    if let Some(f) = &mut lf {
        f.flush().unwrap();
    }
    if let Some(to) = tokens_out {
        let mut all = prompt_ids.clone();
        all.extend_from_slice(&generated);
        std::fs::write(to, as_bytes(&all)).unwrap();
    }
    dump_stats();
    let txt = tok.decode(&generated, false).unwrap_or_default();
    let stdout = std::io::stdout();
    let mut so = stdout.lock();
    let _ = so.write_all(txt.as_bytes());
    if !txt.ends_with('\n') {
        let _ = so.write_all(b"\n");
    }
}
