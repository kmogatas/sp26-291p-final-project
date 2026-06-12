import os
import sys
import ctypes
import numpy as np


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

import compiler


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


def finite_difference(param_name, index, x, y_target, w1, b1, w2, b2, eps=1e-3):
    w1_plus = w1.copy()
    b1_plus = b1.copy()
    w2_plus = w2.copy()
    b2_plus = b2

    w1_minus = w1.copy()
    b1_minus = b1.copy()
    w2_minus = w2.copy()
    b2_minus = b2

    if param_name == "w1":
        w1_plus[index] += eps
        w1_minus[index] -= eps
    elif param_name == "b1":
        b1_plus[index] += eps
        b1_minus[index] -= eps
    elif param_name == "w2":
        w2_plus[index] += eps
        w2_minus[index] -= eps
    elif param_name == "b2":
        b2_plus += eps
        b2_minus -= eps
    else:
        raise ValueError("unknown param_name")

    loss_plus = mlp_loss_numpy(x, y_target, w1_plus, b1_plus, w2_plus, b2_plus)
    loss_minus = mlp_loss_numpy(x, y_target, w1_minus, b1_minus, w2_minus, b2_minus)

    return (loss_plus - loss_minus) / (2.0 * eps)


def main():
    loma_file = os.path.join(os.path.dirname(__file__), "mlp_forward.py")

    with open(loma_file) as f:
        structs, lib = compiler.compile(
            f.read(),
            target="c",
            output_filename="_code/mlp_reverse_gradients"
        )

    lib.mlp_loss.restype = ctypes.c_float

    x = 0.7
    y_target = 0.644217687237691

    w1_np = np.array([0.5, -0.8], dtype=np.float32)
    b1_np = np.array([0.1, 0.2], dtype=np.float32)
    w2_np = np.array([1.2, -0.4], dtype=np.float32)
    b2 = np.float32(0.05)

    FloatArray2 = ctypes.c_float * 2

    w1 = FloatArray2(*w1_np)
    b1 = FloatArray2(*b1_np)
    w2 = FloatArray2(*w2_np)

    dw1 = FloatArray2(0.0, 0.0)
    db1 = FloatArray2(0.0, 0.0)
    dw2 = FloatArray2(0.0, 0.0)
    db2 = ctypes.c_float(0.0)

    dx = ctypes.c_float(0.0)
    dy_target = ctypes.c_float(0.0)

    lib.d_mlp_loss(
        ctypes.c_float(x),
        ctypes.byref(dx),
        ctypes.c_float(y_target),
        ctypes.byref(dy_target),
        w1,
        dw1,
        b1,
        db1,
        w2,
        dw2,
        ctypes.c_float(b2),
        ctypes.byref(db2),
        ctypes.c_float(1.0),
    )

    print("===== Reverse-mode gradients from Loma =====")
    print("dw1:", [dw1[0], dw1[1]])
    print("db1:", [db1[0], db1[1]])
    print("dw2:", [dw2[0], dw2[1]])
    print("db2:", db2.value)

    print("\n===== Finite-difference gradients =====")
    fd_w1_0 = finite_difference("w1", 0, x, y_target, w1_np, b1_np, w2_np, b2)
    fd_w1_1 = finite_difference("w1", 1, x, y_target, w1_np, b1_np, w2_np, b2)
    fd_b1_0 = finite_difference("b1", 0, x, y_target, w1_np, b1_np, w2_np, b2)
    fd_b1_1 = finite_difference("b1", 1, x, y_target, w1_np, b1_np, w2_np, b2)
    fd_w2_0 = finite_difference("w2", 0, x, y_target, w1_np, b1_np, w2_np, b2)
    fd_w2_1 = finite_difference("w2", 1, x, y_target, w1_np, b1_np, w2_np, b2)
    fd_b2 = finite_difference("b2", None, x, y_target, w1_np, b1_np, w2_np, b2)

    print("fd dw1:", [fd_w1_0, fd_w1_1])
    print("fd db1:", [fd_b1_0, fd_b1_1])
    print("fd dw2:", [fd_w2_0, fd_w2_1])
    print("fd db2:", fd_b2)

    print("\n===== Absolute errors =====")
    print("dw1 errors:", [abs(dw1[0] - fd_w1_0), abs(dw1[1] - fd_w1_1)])
    print("db1 errors:", [abs(db1[0] - fd_b1_0), abs(db1[1] - fd_b1_1)])
    print("dw2 errors:", [abs(dw2[0] - fd_w2_0), abs(dw2[1] - fd_w2_1)])
    print("db2 error:", abs(db2.value - fd_b2))


if __name__ == "__main__":
    main()