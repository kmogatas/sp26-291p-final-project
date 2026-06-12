import os
import sys
import ctypes
import numpy as np
import compiler

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)


def softplus_np(x):
    return np.log(1.0 + np.exp(x))


def mlp_loss_numpy(x, y_target, w1, b1, w2, b2):
    z0 = w1[0] * x + b1[0]
    z1 = w1[1] * x + b1[1]

    h0 = softplus_np(z0)
    h1 = softplus_np(z1)

    y_pred = w2[0] * h0 + w2[1] * h1 + b2
    diff = y_pred - y_target
    return diff * diff


def main():
    loma_file = os.path.join(os.path.dirname(__file__), "mlp_forward.py")

    with open(loma_file) as f:
        structs, lib = compiler.compile(
            f.read(),
            target="c",
            output_filename="_code/mlp_forward"
        )

    x = 0.7
    y_target = 0.644217687237691

    w1_np = np.array([0.5, -0.8], dtype=np.float32)
    b1_np = np.array([0.1, 0.2], dtype=np.float32)
    w2_np = np.array([1.2, -0.4], dtype=np.float32)
    b2 = 0.05

    expected = mlp_loss_numpy(
        x,
        y_target,
        w1_np,
        b1_np,
        w2_np,
        b2,
    )

    FloatArray2 = ctypes.c_float * 2

    w1 = FloatArray2(*w1_np)
    b1 = FloatArray2(*b1_np)
    w2 = FloatArray2(*w2_np)

    loma_loss = lib.mlp_loss(
        ctypes.c_float(x),
        ctypes.c_float(y_target),
        w1,
        b1,
        w2,
        ctypes.c_float(b2),
    )

    print("NumPy MLP loss:", expected)
    print("Loma MLP loss:", loma_loss)
    print("Absolute error:", abs(float(loma_loss) - float(expected)))


if __name__ == "__main__":
    main()