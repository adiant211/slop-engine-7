#!/usr/bin/env python3
"""Draw the README charts from bench/results.json (light and dark variants) into docs/."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  blue="#2a78d6", orange="#eb6834", blue_lo="#9ec5f4"),
    "dark": dict(bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 blue="#3987e5", orange="#d95926", blue_lo="#184f95"),
}


def style(ax, t, grid_axis):
    ax.set_facecolor(t["bg"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["left" if grid_axis == "x" else "bottom"].set_visible(True)
    ax.spines["left" if grid_axis == "x" else "bottom"].set_color(t["axis"])
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)


def fmt(v):
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def engine_chart(res, t, path):
    names = ["chat0", "chat1", "chat2", "chat3", "short", "long"]
    P, n = res["prompts"], res["threads"]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=t["bg"])
    style(ax, t, "x")
    h = 0.36
    for i, name in enumerate(names):
        for off, key, col in [(-h / 2 - 0.02, "smol_mt", t["blue"]), (h / 2 + 0.02, "smol_1", t["blue_lo"])]:
            v = P[name][key]["decode_tok_s"]
            ax.barh(i + off, v, height=h, color=col)
            ax.text(v + 4, i + off, fmt(v), va="center", fontsize=9, color=t["ink2"])
    ax.set_yticks(range(len(names)), [f"{m}  ({P[m]['smol_1']['prompt_tokens']} tok)" for m in names])
    ax.tick_params(axis="y", labelcolor=t["ink2"])
    ax.invert_yaxis()
    ax.set_xlabel("generated tokens per second", color=t["muted"], fontsize=9)
    ax.set_xlim(0, max(P[m]["smol_mt"]["decode_tok_s"] for m in names) * 1.12)
    hs = [plt.Rectangle((0, 0), 1, 1, color=t["blue"]), plt.Rectangle((0, 0), 1, 1, color=t["blue_lo"])]
    fig.legend(hs, [f"{n} threads", "1 thread"], loc="upper right", bbox_to_anchor=(0.98, 0.9), frameon=False,
               labelcolor=t["ink2"], fontsize=9, ncols=2)
    fig.text(0.02, 0.95, "Decode speed by prompt", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.02, 0.895, f"128 greedy tokens, prompt name and length on the left  ·  {res['cpu']}", fontsize=9, color=t["muted"])
    fig.subplots_adjust(top=0.82, left=0.19, right=0.97, bottom=0.13)
    fig.savefig(path, dpi=160, facecolor=t["bg"])
    plt.close(fig)


def versus_chart(res, t, path):
    P, n = res["prompts"], res["threads"]
    panels = [
        ("Decode", "tokens / s, higher is better", "chat1", "decode_tok_s", 1),
        ("Prefill (1,005-token prompt)", "tokens / s, higher is better", "long", "prefill_tok_s", 1),
        ("Model load", "milliseconds, lower is better", "chat1", "load_ms", 1),
        ("Whole reply, start to finish", "seconds, lower is better", "chat1", "total_ms", 1e-3),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.9), facecolor=t["bg"])
    w = 0.36
    for ax, (title, unit, prompt, key, k) in zip(axes, panels):
        style(ax, t, "y")
        top = 0
        for g, th in enumerate(["1", "mt"]):
            for off, who, col in [(-w / 2 - 0.02, "torch", t["orange"]), (w / 2 + 0.02, "smol", t["blue"])]:
                v = P[prompt][f"{who}_{th}"][key] * k
                top = max(top, v)
                ax.bar(g + off, v, width=w, color=col)
                ax.annotate(fmt(v), (g + off, v), xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=9, color=t["ink2"])
        ax.set_ylim(0, top * 1.14)
        ax.set_xticks([0, 1], ["1 thread", f"{n} threads"])
        ax.tick_params(axis="x", labelcolor=t["ink2"])
        ax.set_title(title, fontsize=10, fontweight="bold", color=t["ink"], loc="left", pad=20)
        ax.text(0, 1.035, unit, transform=ax.transAxes, fontsize=8.5, color=t["muted"])
    hs = [plt.Rectangle((0, 0), 1, 1, color=t["orange"]), plt.Rectangle((0, 0), 1, 1, color=t["blue"])]
    fig.legend(hs, [f"PyTorch {res['torch'].split('+')[0]} (fp32)", "this engine"], loc="upper right",
               bbox_to_anchor=(0.985, 0.985), frameon=False, labelcolor=t["ink2"], fontsize=9, ncols=2)
    fig.text(0.015, 0.94, "PyTorch vs this engine", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.015, 0.885, f"SmolLM2-135M-Instruct, 128 greedy tokens, median of {res['runs']} runs  ·  {res['cpu']}",
             fontsize=9, color=t["muted"])
    fig.subplots_adjust(top=0.70, left=0.04, right=0.985, bottom=0.09, wspace=0.28)
    fig.savefig(path, dpi=160, facecolor=t["bg"])
    plt.close(fig)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "bench/results.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "docs"
    res = json.load(open(src))
    os.makedirs(out, exist_ok=True)
    for name, t in THEMES.items():
        engine_chart(res, t, os.path.join(out, f"bench-{name}.png"))
        versus_chart(res, t, os.path.join(out, f"torch-vs-smol-{name}.png"))


if __name__ == "__main__":
    main()
