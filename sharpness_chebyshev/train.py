import argparse
from typing import List, Sequence, Tuple

import torch


def generate_dataset(num_points: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate the noisy seventh-order Chebyshev dataset."""
    x = torch.linspace(-1.0, 0.99, steps=num_points).unsqueeze(1)
    noise = torch.rand_like(x) * 0.5
    y = 64 * x ** 7 - 112 * x ** 5 + 56 * x ** 3 - 7 * x + noise
    return x, y


def compute_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Compute mean squared error loss for the current parameters."""
    hidden = torch.tanh(x * w + b)
    preds = hidden @ v.unsqueeze(1) + c
    return torch.mean((preds - y) ** 2)


def flatten_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def largest_eigenvalue(matrix: torch.Tensor) -> float:
    """Return the largest real part eigenvalue of a symmetric matrix."""
    eigvals = torch.linalg.eigvals(matrix)
    return eigvals.real.max().item()


def train(
    steps: int = 100_000,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    hessian_interval: int = 1,
) -> None:
    torch.manual_seed(seed)

    x, y = generate_dataset()
    x = x.to(device)
    y = y.to(device)

    w = torch.randn(5, device=device, requires_grad=True)
    b = torch.zeros(5, device=device, requires_grad=True)
    v = torch.randn(5, device=device, requires_grad=True)
    c = torch.zeros(1, device=device, requires_grad=True)

    params = [w, b, v, c]

    losses: List[float] = []
    sharpness: List[float] = []

    for step in range(steps):
        loss = compute_loss(x, y, w, b, v, c)
        losses.append(loss.item())

        if step % hessian_interval == 0:
            grads = torch.autograd.grad(loss, params, create_graph=True)
            flat_grads = flatten_tensors(grads)

            num_params = flat_grads.numel()
            hessian_rows: List[torch.Tensor] = []
            for i in range(num_params):
                second_grads = torch.autograd.grad(
                    flat_grads[i], params, retain_graph=True, allow_unused=False
                )
                hessian_rows.append(flatten_tensors(second_grads))

            hessian = torch.stack(hessian_rows)
            sharpness.append(largest_eigenvalue(hessian))

            grads_for_update = [g.detach() for g in grads]
            del grads, flat_grads, hessian_rows, hessian
        else:
            grads_for_update = torch.autograd.grad(loss, params)
            sharpness.append(sharpness[-1])

        with torch.no_grad():
            for param, grad in zip(params, grads_for_update):
                param -= lr * grad

    steps_tensor = torch.arange(steps, dtype=torch.int64)

    def write_two_column_file(
        path: str, left_header: str, right_header: str, left: Sequence, right: Sequence
    ) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"{left_header}\t{right_header}\n")
            for left_value, right_value in zip(left, right):
                handle.write(f"{int(left_value)}\t{float(right_value)}\n")

    write_two_column_file(
        "loss_vs_step.tsv", "step", "loss", steps_tensor.tolist(), losses
    )
    write_two_column_file(
        "sharpness_vs_step.tsv",
        "step",
        "sharpness",
        steps_tensor.tolist(),
        sharpness,
    )
    print("Saved training statistics to loss_vs_step.tsv and sharpness_vs_step.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chebyshev sharpness model")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--hessian-interval",
        type=int,
        default=1,
        help="Frequency (in steps) for computing the Hessian. Must divide steps.",
    )
    args = parser.parse_args()

    train(
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        hessian_interval=max(1, args.hessian_interval),
    )
