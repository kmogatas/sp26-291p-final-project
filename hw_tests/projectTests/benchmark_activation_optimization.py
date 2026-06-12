import os
import sys
import ctypes
import time
import numpy as np
import compiler

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)


def compile_loma(filename, output_name):
    loma_file = os.path.join(os.path.dirname(__file__), filename)
    with open(loma_file) as f:
        structs, lib = compiler.compile(
            f.read(),
            target="c",
            output_filename=output_name
        )
    lib.mlp_loss.restype = ctypes.c_float
    return lib


def run_training(lib, steps=1000):
    FloatArray2 = ctypes.c_float * 2

    x = 0.7
    y_target = np.sin(x)

    w1_np = np.array([0.5, -0.8], dtype=np.float32)
    b1_np = np.array([0.1, 0.2], dtype=np.float32)
    w2_np = np.array([1.2, -0.4], dtype=np.float32)
    b2_np = np.float32(0.05)

    lr = 0.05
    final_loss = None

    start = time.perf_counter()

    for step in range(steps):
        w1 = FloatArray2(*w1_np)
        b1 = FloatArray2(*b1_np)
        w2 = FloatArray2(*w2_np)

        dw1 = FloatArray2(0.0, 0.0)
        db1 = FloatArray2(0.0, 0.0)
        dw2 = FloatArray2(0.0, 0.0)
        db2 = ctypes.c_float(0.0)

        dx = ctypes.c_float(0.0)
        dy_target = ctypes.c_float(0.0)

        final_loss = lib.mlp_loss(
            ctypes.c_float(x),
            ctypes.c_float(y_target),
            w1,
            b1,
            w2,
            ctypes.c_float(b2_np),
        )

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
            ctypes.c_float(b2_np),
            ctypes.byref(db2),
            ctypes.c_float(1.0),
        )

        w1_np[0] -= lr * dw1[0]
        w1_np[1] -= lr * dw1[1]
        b1_np[0] -= lr * db1[0]
        b1_np[1] -= lr * db1[1]
        w2_np[0] -= lr * dw2[0]
        w2_np[1] -= lr * dw2[1]
        b2_np -= lr * db2.value

    end = time.perf_counter()

    return final_loss, end - start


def main():
    softplus_lib = compile_loma(
        "mlp_forward.py",
        "_code/mlp_softplus_benchmark"
    )

    relu_lib = compile_loma(
        "mlp_forward_relu.py",
        "_code/mlp_relu_benchmark"
    )

    steps = 1000

    softplus_loss, softplus_time = run_training(softplus_lib, steps)
    relu_loss, relu_time = run_training(relu_lib, steps)

    print("===== Activation optimization benchmark =====")
    print("Steps:", steps)
    print()
    print("Softplus final loss:", softplus_loss)
    print("Softplus time:", softplus_time)
    print()
    print("ReLU final loss:", relu_loss)
    print("ReLU time:", relu_time)
    print()
    print("Speedup softplus_time / relu_time:", softplus_time / relu_time)


if __name__ == "__main__":
    main()