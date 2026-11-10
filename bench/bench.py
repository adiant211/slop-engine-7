#!/usr/bin/env python3
"""Time the engine against PyTorch (fp32, transformers) on the same machine and prompts.

    python bench/bench.py --smol target/release/smol --model model --out bench/results.json

Needs torch, transformers and tokenizers. Both sides generate 128 greedy tokens with a key/value cache,
on 1 thread and on --threads threads; each number is the median of --runs runs. The torch "load" time is from_pretrained plus the tokenizer and
does not count `import torch`; the engine's load time is the whole of its start-up.
"""
import argparse, json, os, platform, re, statistics, struct, subprocess, tempfile, time

CHAT = ("<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
N = 128


def med(xs):
    return round(statistics.median(xs), 2)


def run_smol(args, prompt, chat, threads):
    env = dict(os.environ, SMOL_STATS="1")
    cmd = [args.smol, "gen", "--model", args.model, "--prompt-file", prompt, "--n", str(N)]
    if chat:
        cmd.append("--chat")
    env["SMOL_NT"] = str(threads)
    if threads == 1:
        cmd = ["taskset", "-c", "2"] + cmd
    out = tempfile.NamedTemporaryFile(suffix=".u32", delete=False).name
    rows = []
    for _ in range(args.runs):
        t = time.perf_counter()
        p = subprocess.run(cmd + ["--tokens-out", out], env=env, capture_output=True, text=True, check=True)
        wall = (time.perf_counter() - t) * 1e3
        s = dict(kv.split("=") for kv in re.search(r"SMOL_STATS (.*)", p.stderr).group(1).split())
        rows.append((float(s["load_ms"]), float(s["prefill_ms"]), float(s["decode_ms"]), wall,
                     int(s["prompt_tokens"]), int(s["gen_tokens"])))
    raw = open(out, "rb").read()
    os.unlink(out)
    toks = list(struct.unpack("<%dI" % (len(raw) // 4), raw))
    plen, glen = rows[0][4], rows[0][5]
    return {
        "load_ms": med([r[0] for r in rows]),
        "prefill_tok_s": med([plen / r[1] * 1e3 for r in rows]),
        "decode_tok_s": med([glen / r[2] * 1e3 for r in rows]),
        "total_ms": med([r[3] for r in rows]),
        "prompt_tokens": plen,
    }, toks[-N:]


def run_torch(args, text, threads):
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(threads)
    rows = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
        t1 = time.perf_counter()
        ids = torch.tensor([tok.encode(text, add_special_tokens=False).ids])
        with torch.inference_mode():
            o = model(ids, use_cache=True)
            nxt = o.logits[0, -1].argmax().view(1, 1)
            t2 = time.perf_counter()
            gen = [nxt.item()]
            for _ in range(N - 1):
                o = model(nxt, past_key_values=o.past_key_values, use_cache=True)
                nxt = o.logits[0, -1].argmax().view(1, 1)
                gen.append(nxt.item())
        t3 = time.perf_counter()
        rows.append((t1 - t0, ids.shape[1] / (t2 - t1), (N - 1) / (t3 - t2), t3 - t0))
        del model
    return {
        "load_ms": med([r[0] * 1e3 for r in rows]),
        "prefill_tok_s": med([r[1] for r in rows]),
        "decode_tok_s": med([r[2] for r in rows]),
        "total_ms": med([r[3] * 1e3 for r in rows]),
        "prompt_tokens": int(ids.shape[1]),
    }, gen


def accuracy(args, ctx=1024):
    """Engine logits against torch fp32 logits on the first `ctx` tokens of the dev text."""
    import torch
    from transformers import AutoModelForCausalLM
    raw = open(args.dev, "rb").read()[:ctx * 4]
    tin = tempfile.NamedTemporaryFile(suffix=".u32", delete=False).name
    out = tin + ".f32"
    open(tin, "wb").write(raw)
    subprocess.run([args.smol, "ppl", "--model", args.model, "--tokens", tin, "--ctx", str(ctx), "--logits", out],
                   capture_output=True, check=True)
    ids = torch.tensor([list(struct.unpack("<%dI" % ctx, raw))])
    ours = torch.frombuffer(bytearray(open(out, "rb").read()), dtype=torch.float32).view(ctx, -1)
    os.unlink(tin), os.unlink(out)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    with torch.inference_mode():
        ref = model(ids).logits[0]
    lo, lr = ours.log_softmax(-1), ref.log_softmax(-1)
    nxt = ids[0, 1:, None]
    return {
        "tokens": ctx,
        "ppl_torch": round(float((-lr[:-1].gather(1, nxt).mean()).exp()), 3),
        "ppl_smol": round(float((-lo[:-1].gather(1, nxt).mean()).exp()), 3),
        "top1_agreement": round(float((ours.argmax(-1) == ref.argmax(-1)).float().mean()), 4),
        "mean_kld": round(float((lr.exp() * (lr - lo)).sum(-1).mean()), 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smol", default="target/release/smol")
    ap.add_argument("--model", default="model")
    ap.add_argument("--prompts", default="data/prompts")
    ap.add_argument("--out", default="bench/results.json")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4, help="thread count of the multi-thread runs")
    ap.add_argument("--dev", default="data/dev.u32")
    args = ap.parse_args()

    import torch, transformers
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    cpu = next((l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
    res = {"cpu": cpu, "torch": torch.__version__, "transformers": transformers.__version__,
           "tokens": N, "runs": args.runs, "threads": args.threads, "prompts": {}}
    for name in ["chat0", "chat1", "chat2", "chat3", "short", "long"]:
        path = os.path.join(args.prompts, name + ".txt")
        chat = name.startswith("chat")
        text = open(path, newline="").read()
        text = CHAT.format(text) if chat else text
        entry = {}
        for label, threads in [("1", 1), ("mt", args.threads)]:
            entry["smol_" + label], smol_toks = run_smol(args, path, chat, threads)
            if name in ("chat1", "long"):
                entry["torch_" + label], torch_toks = run_torch(args, text, threads)
                same = next((i for i, (a, b) in enumerate(zip(smol_toks, torch_toks)) if a != b), N)
                entry["same_leading_tokens_" + label] = same
        res["prompts"][name] = entry
        print(name, json.dumps(entry), flush=True)
    res["accuracy"] = accuracy(args)
    print("accuracy", json.dumps(res["accuracy"]), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
