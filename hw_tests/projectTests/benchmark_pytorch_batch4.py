import time
import numpy as np
import torch


def train_pytorch_batch4(steps=1000):
    xs = torch.tensor([-1.0, -0.25, 0.25, 1.0], dtype=torch.float32)
    ys = torch.sin(xs)

    w1 = torch.tensor([0.5, -0.8], dtype=torch.float32, requires_grad=True)
    b1 = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)
    w2 = torch.tensor([1.2, -0.4], dtype=torch.float32, requires_grad=True)
    b2 = torch.tensor(0.05, dtype=torch.float32, requires_grad=True)

    lr = 0.05
    final_loss = None

    start = time.perf_counter()

    for _ in range(steps):
        z0 = w1[0] * xs + b1[0]
        z1 = w1[1] * xs + b1[1]

        h0 = torch.log(1.0 + torch.exp(z0))
        h1 = torch.log(1.0 + torch.exp(z1))

        y_pred = w2[0] * h0 + w2[1] * h1 + b2
        loss = torch.mean((y_pred - ys) * (y_pred - ys))

        loss.backward()

        with torch.no_grad():
            w1 -= lr * w1.grad
            b1 -= lr * b1.grad
            w2 -= lr * w2.grad
            b2 -= lr * b2.grad

            w1.grad.zero_()
            b1.grad.zero_()
            w2.grad.zero_()
            b2.grad.zero_()

        final_loss = loss.item()

    end = time.perf_counter()
    return final_loss, end - start


def main():
    steps = 1000

    # Warmup to reduce one-time PyTorch overhead.
    train_pytorch_batch4(steps=10)

    loss, elapsed = train_pytorch_batch4(steps=steps)

    print("===== PyTorch batch-4 benchmark =====")
    print("Steps:", steps)
    print("Batch size:", 4)
    print("PyTorch batch final loss:", loss)
    print("PyTorch batch time:", elapsed)


if __name__ == "__main__":
    main()