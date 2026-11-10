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

Every message runs `smol gen` (a few rounds with a growing token limit until the model ends its turn), so a
reply costs a fraction of a second of model loading plus generation at roughly 100-300 tokens per second.

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
