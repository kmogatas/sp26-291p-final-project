import time
import numpy as np
import torch


def train_pytorch_softplus(steps=1000):
    x = torch.tensor(0.7, dtype=torch.float32)
    y_target = torch.sin(x)

    w1 = torch.tensor([0.5, -0.8], dtype=torch.float32, requires_grad=True)
    b1 = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)
    w2 = torch.tensor([1.2, -0.4], dtype=torch.float32, requires_grad=True)
    b2 = torch.tensor(0.05, dtype=torch.float32, requires_grad=True)

    lr = 0.05
    final_loss = None

    start = time.perf_counter()

    for _ in range(steps):
        z = w1 * x + b1
        h = torch.log(1.0 + torch.exp(z))
        y_pred = torch.sum(w2 * h) + b2
        loss = (y_pred - y_target) * (y_pred - y_target)

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


def train_pytorch_relu(steps=1000):
    x = torch.tensor(0.7, dtype=torch.float32)
    y_target = torch.sin(x)

    w1 = torch.tensor([0.5, -0.8], dtype=torch.float32, requires_grad=True)
    b1 = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)
    w2 = torch.tensor([1.2, -0.4], dtype=torch.float32, requires_grad=True)
    b2 = torch.tensor(0.05, dtype=torch.float32, requires_grad=True)

    lr = 0.05
    final_loss = None

    start = time.perf_counter()

    for _ in range(steps):
        z = w1 * x + b1
        h = torch.relu(z)
        y_pred = torch.sum(w2 * h) + b2
        loss = (y_pred - y_target) * (y_pred - y_target)

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

    train_pytorch_softplus(steps=10)
    train_pytorch_relu(steps=10)

    softplus_loss, softplus_time = train_pytorch_softplus(steps)
    relu_loss, relu_time = train_pytorch_relu(steps)

    print("===== PyTorch comparison benchmark =====")
    print("Steps:", steps)
    print()
    print("PyTorch Softplus final loss:", softplus_loss)
    print("PyTorch Softplus time:", softplus_time)
    print()
    print("PyTorch ReLU final loss:", relu_loss)
    print("PyTorch ReLU time:", relu_time)
    print()
    print("PyTorch softplus_time / relu_time:", softplus_time / relu_time)


if __name__ == "__main__":
    main()