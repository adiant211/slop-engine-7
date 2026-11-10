# SmolLM2-135M-Instruct CPU engine

A from-scratch CPU inference engine for SmolLM2-135M-Instruct in Rust (std plus `tokenizers`, `half`,
`safetensors`, `gemm`). Weights are read from the model directory on every run; nothing is cached.

## Try it in a browser

```bash
./run.sh
```

then open <http://127.0.0.1:8000/>. The script checks the CPU (AVX2, FMA, F16C), downloads
`model.safetensors`, `tokenizer.json` and `config.json` from
[HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
into `./model` (about 270 MB, skipped when already there), builds with `cargo build --release`, and starts
`smol-ui`, a std-only web server (`src/bin/smol-ui.rs`, page in `ui/index.html`).

Settings by environment variable: `PORT`, `HOST` (default `127.0.0.1`; the page has no login), `MODEL_DIR`,
`HF_REPO`, `HF_ENDPOINT`, `HF_TOKEN`. Needs Linux x86-64 and a Rust toolchain; on Windows use WSL.

Every message is one `smol gen --stop` run (the engine stops by itself at the end-of-turn token), so a reply
costs about 0.1 s of model and tokenizer loading plus generation at roughly 200 tokens per second on one core
and 300+ on several.

Under each reply the page prints its speed: prompt tokens and prefill tok/s, reply tokens and decode tok/s,
model load time, total time (and how many engine runs it took), and the end-to-end rate (reply tokens over total
time). The numbers come from `SMOL_STATS=1`, which makes `smol gen` print one line of per-phase timings on
stderr; the server sets it and sends the result in an `X-Stats` response header.

## Command line

```bash
cargo build --release
target/release/smol gen --model model --prompt-file prompt.txt --chat --n 128
target/release/smol ppl --model model --tokens in.u32 --ctx 1024 --logits out.f32
```

`gen` writes greedy tokens (ties to the lowest id); `--logits` and `--tokens-out` do not change the tokens.
`--stop` ends the reply at the first end-of-turn or end-of-text token. `SMOL_NT=1` runs on one core.

## What a single query pays for

- Loading: the weight file is mapped (not copied) and quantised on the fly; the low-byte plane is transposed
  16 rows at a time, skipped for the matrix that never uses it, and made row by row on first use for the head.
  The tokenizer loads on its own thread meanwhile.
- Decoding is bound by memory bandwidth (every token streams all the weights once), so the output head, a
  fifth of those bytes, is read at 4 bits per weight when only the greedy token is wanted: that first pass
  brackets every logit, and the few hundred rows that can still win are then computed exactly as before.
  `--logits` and `ppl` always use the full head; `SMOL_HEAD=full` forces it for `gen` too.
