import argparse
from pathlib import Path

import torch

from model import TransformerConfig, TinyTransformerLM


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join([itos[i] for i in ids])
    return stoi, itos, encode, decode


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    idx = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    # x is a chunk of tokens; y is x shifted by one token (next-token labels).
    x = torch.stack([data[i : i + block_size] for i in idx]).to(device)
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in idx]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device):
    # Evaluation disables dropout and gradient tracking.
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    default_data = Path(__file__).resolve().parents[2] / "nanoGPT/data/input.txt"
    parser = argparse.ArgumentParser(description="Train a tiny base Transformer language model")
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    # Toggle GQA on/off
    parser.add_argument("--use-gqa", action="store_true")
    # Number of KV heads when GQA is enabled (must divide n_heads)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = load_text(args.data)
    stoi, itos, encode, decode = build_vocab(text)
    # Character-level tokenization for maximal transparency.
    data = torch.tensor(encode(text), dtype=torch.long)

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    cfg = TransformerConfig(
        vocab_size=len(stoi),
        block_size=args.block_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_gqa=args.use_gqa,
        n_kv_heads=args.n_kv_heads,
    )
    model = TinyTransformerLM(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"device={device}, params={sum(p.numel() for p in model.parameters()):,}")
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.block_size,
                args.batch_size,
                args.eval_iters,
                device,
            )
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)
        # Standard update: zero grads -> backprop -> clip -> optimizer step.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=300)[0].tolist()
    print("\n--- sample ---")
    print(decode(generated))


if __name__ == "__main__":
    main()
