"""第 10 课 · 用训好的 GPT 续写。

运行: python sample.py [--temperature 0.8] [--top-k 40] [--tokens 500]
      python sample.py --prompt "ROMEO:" --temperature 0.8
"""

import argparse
import os

import torch

from model import GPT, GPTConfig

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--tokens", type=int, default=500)
    ap.add_argument("--prompt", type=str, default="")
    args = ap.parse_args()

    ckpt = torch.load(os.path.join(ROOT, "model.pt"), weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    chars = ckpt["vocab"]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.prompt:
        idx = torch.tensor([[stoi[c] for c in args.prompt]])
    else:
        idx = torch.zeros((1, 1), dtype=torch.long)

    out = model.generate(idx, args.tokens,
                         temperature=args.temperature, top_k=args.top_k)
    print("".join(itos[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
