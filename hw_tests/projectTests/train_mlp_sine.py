import os
import sys
import ctypes
import numpy as np
import compiler

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)


def main():
    loma_file = os.path.join(os.path.dirname(__file__), "mlp_forward.py")

    with open(loma_file) as f:
        structs, lib = compiler.compile(
            f.read(),
            target="c",
            output_filename="_code/train_mlp_sine"
        )

    lib.mlp_loss.restype = ctypes.c_float

    FloatArray2 = ctypes.c_float * 2

    x = 0.7
    y_target = np.sin(x)

    w1_np = np.array([0.5, -0.8], dtype=np.float32)
    b1_np = np.array([0.1, 0.2], dtype=np.float32)
    w2_np = np.array([1.2, -0.4], dtype=np.float32)
    b2_np = np.float32(0.05)

    lr = 0.05

    for step in range(100):
        w1 = FloatArray2(*w1_np)
        b1 = FloatArray2(*b1_np)
        w2 = FloatArray2(*w2_np)

        dw1 = FloatArray2(0.0, 0.0)
        db1 = FloatArray2(0.0, 0.0)
        dw2 = FloatArray2(0.0, 0.0)
        db2 = ctypes.c_float(0.0)

        dx = ctypes.c_float(0.0)
        dy_target = ctypes.c_float(0.0)

        loss = lib.mlp_loss(
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

        if step % 10 == 0:
            print("step:", step, "loss:", loss)

    print("Final weights:")
    print("w1:", w1_np)
    print("b1:", b1_np)
    print("w2:", w2_np)
    print("b2:", b2_np)


if __name__ == "__main__":
    main()