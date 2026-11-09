#!/usr/bin/env bash
# Install and run: downloads the model from Hugging Face, builds the engine, starts the web UI.
#
#   ./run.sh                      then open http://127.0.0.1:8000/
#   PORT=9000 ./run.sh            another port
#   HOST=0.0.0.0 ./run.sh         reachable from other machines (there is no login: only on a trusted network)
#   MODEL_DIR=/data/smollm ./run.sh
#   HF_TOKEN=... ./run.sh         only if you mirror the model somewhere private
#
# Needs: Linux x86-64 with AVX2/FMA/F16C (on Windows use WSL), curl, and a Rust toolchain (https://rustup.rs).
set -euo pipefail
cd "$(dirname "$0")"

REPO=${HF_REPO:-HuggingFaceTB/SmolLM2-135M-Instruct}
HF=${HF_ENDPOINT:-https://huggingface.co}
MODEL_DIR=${MODEL_DIR:-model}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}

die() { echo "run.sh: $*" >&2; exit 1; }

[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || die "this engine runs on Linux x86-64 only (on Windows, use WSL)"
for flag in avx2 fma f16c; do
  grep -qw "$flag" /proc/cpuinfo || die "this CPU has no $flag, which the engine needs"
done
command -v curl >/dev/null || die "curl is not installed"
command -v cargo >/dev/null || die "no Rust toolchain found; install one with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"

mkdir -p "$MODEL_DIR"
auth=()
[ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")

fetch() {
  local f=$1 dst="$MODEL_DIR/$1"
  [ -s "$dst" ] && return 0
  echo "downloading $f from $REPO"
  curl -fL --retry 3 -C - ${auth[@]+"${auth[@]}"} -o "$dst.part" "$HF/$REPO/resolve/main/$f" ||
    { rm -f "$dst.part"; curl -fL --retry 3 ${auth[@]+"${auth[@]}"} -o "$dst.part" "$HF/$REPO/resolve/main/$f"; } ||
    die "download of $f failed"
  mv "$dst.part" "$dst"
}
fetch tokenizer.json
fetch config.json
fetch model.safetensors
size=$(wc -c < "$MODEL_DIR/model.safetensors")
[ "$size" -gt 200000000 ] || { rm -f "$MODEL_DIR/model.safetensors"; die "model.safetensors is only $size bytes (interrupted download?); run again"; }

echo "building (first time takes a minute or two)..."
cargo build --release --quiet

exec target/release/smol-ui --model "$MODEL_DIR" --host "$HOST" --port "$PORT"
