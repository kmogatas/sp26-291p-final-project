import numpy as np


def softplus(x):
    return np.log(1.0 + np.exp(x))


def mlp_loss_numpy(x, y_target, w1, b1, w2, b2):
    z0 = w1[0] * x + b1[0]
    z1 = w1[1] * x + b1[1]

    h0 = softplus(z0)
    h1 = softplus(z1)

    y_pred = w2[0] * h0 + w2[1] * h1 + b2

    diff = y_pred - y_target
    loss = diff * diff

    return loss


def main():
    x = 0.7
    y_target = 0.644217687237691
    w1 = [0.5, -0.8]
    b1 = [0.1, 0.2]
    w2 = [1.2, -0.4]
    b2 = 0.05
    loss = mlp_loss_numpy(x, y_target, w1, b1, w2, b2)

    print("x:", x)
    print("y_target:", y_target)
    print("NumPy MLP loss:", loss)


if __name__ == "__main__":
    main()