// Small web UI for the smol engine. std only: serves one page and runs `smol gen` for every message.
//
//   smol-ui [--model DIR] [--host ADDR] [--port N] [--engine PATH]
//
// GET  /          the chat page (ui/index.html, compiled in)
// POST /gen?n=N   body = a complete ChatML prompt (the page builds it) -> the assistant's reply as plain text,
//                 with an X-Stats header: token counts and the time of each phase (see `generate`)
//
// The engine always produces exactly N tokens and does not stop at the end-of-turn token, so the reply is
// generated with a small N first and N is doubled (up to the limit) until <|im_end|> shows up. Greedy decoding
// is deterministic, so every round extends the previous one.
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const PAGE: &str = include_str!("../../ui/index.html");
const MAX_BODY: usize = 1 << 20;
const MAX_TOKENS: usize = 1024;
// SmolLM2 special tokens that end a reply: <|endoftext|> and <|im_end|>
const STOP_IDS: [u32; 2] = [0, 2];
static COUNTER: AtomicUsize = AtomicUsize::new(0);

struct Config {
    engine: PathBuf,
    model: String,
}

fn main() {
    let mut model = String::from("model");
    let mut host = String::from("127.0.0.1");
    let mut port: u16 = 8000;
    let mut engine: Option<PathBuf> = None;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let val = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--model" => model = val,
            "--host" => host = val,
            "--port" => port = val.parse().unwrap_or_else(|_| die("bad --port")),
            "--engine" => engine = Some(PathBuf::from(val)),
            _ => die("usage: smol-ui [--model DIR] [--host ADDR] [--port N] [--engine PATH]"),
        }
        i += 2;
    }
    let engine = engine.unwrap_or_else(|| {
        let exe = std::env::current_exe().unwrap_or_else(|_| die("cannot locate this program"));
        exe.parent().unwrap().join("smol")
    });
    if !engine.is_file() {
        die(&format!("engine not found: {} (run `cargo build --release`)", engine.display()));
    }
    for f in ["model.safetensors", "tokenizer.json"] {
        if !PathBuf::from(&model).join(f).is_file() {
            die(&format!("{}/{} is missing (run ./run.sh to download the model)", model, f));
        }
    }
    let listener = TcpListener::bind((host.as_str(), port)).unwrap_or_else(|e| die(&format!("cannot listen on {}:{}: {}", host, port, e)));
    println!("smol-ui: http://{}:{}/   (model {}, engine {})", host, port, model, engine.display());
    let cfg = Arc::new(Config { engine, model });
    let gate = Arc::new(Mutex::new(())); // one generation at a time: the engine uses every core
    for stream in listener.incoming().flatten() {
        let (cfg, gate) = (cfg.clone(), gate.clone());
        std::thread::spawn(move || {
            let _ = handle(stream, &cfg, &gate);
        });
    }
}

fn die(msg: &str) -> ! {
    eprintln!("smol-ui: {}", msg);
    std::process::exit(2)
}

fn reply(s: &mut TcpStream, code: u16, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    reply_with(s, code, ctype, "", body)
}

// extra: more header lines, each ending in \r\n
fn reply_with(s: &mut TcpStream, code: u16, ctype: &str, extra: &str, body: &[u8]) -> std::io::Result<()> {
    let status = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        _ => "Internal Server Error",
    };
    write!(s, "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\n{}Connection: close\r\n\r\n", code, status, ctype, body.len(), extra)?;
    s.write_all(body)?;
    s.flush()
}

fn handle(mut s: TcpStream, cfg: &Config, gate: &Mutex<()>) -> std::io::Result<()> {
    s.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut r = BufReader::new(s.try_clone()?);
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(());
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();
    let mut clen = 0usize;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                clen = v.trim().parse().unwrap_or(0);
            }
        }
    }
    if clen > MAX_BODY {
        return reply(&mut s, 413, "text/plain", b"prompt too long");
    }
    let mut body = vec![0u8; clen];
    r.read_exact(&mut body)?;
    let (path, query) = target.split_once('?').unwrap_or((target.as_str(), ""));
    match (method.as_str(), path) {
        ("GET", "/") | ("GET", "/index.html") => reply(&mut s, 200, "text/html; charset=utf-8", PAGE.as_bytes()),
        ("POST", "/gen") => {
            let n = query
                .split('&')
                .find_map(|kv| kv.strip_prefix("n="))
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(256)
                .clamp(1, MAX_TOKENS);
            let _turn = gate.lock().unwrap_or_else(|e| e.into_inner());
            match generate(cfg, &body, n) {
                Ok(g) => {
                    let stats = format!(
                        "X-Stats: prompt={};gen={};rounds={};decode_tokens={};load_ms={:.1};prefill_ms={:.1};decode_ms={:.1};total_ms={:.1}\r\n",
                        g.prompt, g.gen, g.rounds, g.decode_tokens, g.load_ms, g.prefill_ms, g.decode_ms, g.total_ms
                    );
                    reply_with(&mut s, 200, "text/plain; charset=utf-8", &stats, g.text.as_bytes())
                }
                Err(e) => reply(&mut s, 500, "text/plain; charset=utf-8", e.as_bytes()),
            }
        }
        _ => reply(&mut s, 404, "text/plain", b"not found"),
    }
}

// What one request cost. The phase times come from the last engine run (it ran the prompt once and then
// decoded `decode_tokens` tokens, of which `gen` belong to the reply); `total_ms` covers every round.
struct Generated {
    text: String,
    prompt: usize,      // prompt tokens
    gen: usize,         // reply tokens (up to the end-of-turn token)
    decode_tokens: usize, // tokens the last engine run decoded
    rounds: usize,
    load_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    total_ms: f64,
}

fn stat(line: &str, key: &str) -> f64 {
    line.split_whitespace()
        .find_map(|kv| kv.strip_prefix(key).and_then(|v| v.strip_prefix('=')))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0)
}

fn generate(cfg: &Config, prompt: &[u8], max: usize) -> Result<Generated, String> {
    let started = Instant::now();
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp = std::env::temp_dir();
    let file = tmp.join(format!("smol-ui-{}-{}.txt", std::process::id(), id));
    let toks = tmp.join(format!("smol-ui-{}-{}.tok", std::process::id(), id));
    std::fs::write(&file, prompt).map_err(|e| format!("cannot write prompt file: {}", e))?;
    // one engine run: it stops by itself at the end-of-turn token (--stop), or after `max` tokens
    let rounds = 1;
    let result = (|| {
        let out = Command::new(&cfg.engine)
            .args(["gen", "--model", &cfg.model, "--prompt-file"])
            .arg(&file)
            .args(["--stop", "--n", &max.to_string(), "--tokens-out"])
            .arg(&toks)
            .env("SMOL_STATS", "1")
            .output()
            .map_err(|e| format!("cannot run the engine: {}", e))?;
        let err = String::from_utf8_lossy(&out.stderr).into_owned();
        if !out.status.success() {
            return Err(format!("engine failed: {}", err.lines().last().unwrap_or("")));
        }
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        let stop = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|m| text.find(m)).min();
        let text = match stop {
            Some(at) => text[..at].trim().to_string(),
            None => text.trim_end().to_string(),
        };
        let line = err.lines().find(|l| l.starts_with("SMOL_STATS")).unwrap_or("");
        // token counts from the engine: the prompt, then the tokens it decoded (the stop token last)
        let (plen, n) = (stat(line, "prompt_tokens") as usize, stat(line, "gen_tokens") as usize);
        let ids: Vec<u32> = std::fs::read(&toks)
            .unwrap_or_default()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let tail = &ids[ids.len().saturating_sub(n)..];
        let gen = tail.iter().position(|t| STOP_IDS.contains(t)).unwrap_or(n);
        Ok(Generated {
            text,
            prompt: plen,
            gen,
            decode_tokens: n,
            rounds,
            load_ms: stat(line, "load_ms"),
            prefill_ms: stat(line, "prefill_ms"),
            decode_ms: stat(line, "decode_ms"),
            total_ms: 0.0,
        })
    })();
    let _ = std::fs::remove_file(&file);
    let _ = std::fs::remove_file(&toks);
    result.map(|mut g| {
        g.total_ms = started.elapsed().as_secs_f64() * 1e3;
        g
    })
}
