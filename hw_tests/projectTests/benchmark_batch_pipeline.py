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
    return lib


def run_per_sample_training(lib, xs_np, ys_np, steps=1000):
    FloatArray2 = ctypes.c_float * 2

    w1_np = np.array([0.5, -0.8], dtype=np.float32)
    b1_np = np.array([0.1, 0.2], dtype=np.float32)
    w2_np = np.array([1.2, -0.4], dtype=np.float32)
    b2_np = np.float32(0.05)

    lr = 0.05
    final_loss = 0.0

    lib.mlp_loss.restype = ctypes.c_float

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

        total_loss = 0.0

        #batch
        for i in range(4):
            x = float(xs_np[i])
            y_target = float(ys_np[i])

            sample_loss = lib.mlp_loss(
                ctypes.c_float(x),
                ctypes.c_float(y_target),
                w1,
                b1,
                w2,
                ctypes.c_float(b2_np),
            )

            total_loss += sample_loss

            
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
                ctypes.c_float(0.25),
            )

        final_loss = total_loss / 4.0

        w1_np[0] -= lr * dw1[0]
        w1_np[1] -= lr * dw1[1]
        b1_np[0] -= lr * db1[0]
        b1_np[1] -= lr * db1[1]
        w2_np[0] -= lr * dw2[0]
        w2_np[1] -= lr * dw2[1]
        b2_np -= lr * db2.value

    end = time.perf_counter()

    return final_loss, end - start


def run_batch_training(lib, xs_np, ys_np, steps=1000):
    FloatArray2 = ctypes.c_float * 2
    FloatArray4 = ctypes.c_float * 4

    w1_np = np.array([0.5, -0.8], dtype=np.float32)
    b1_np = np.array([0.1, 0.2], dtype=np.float32)
    w2_np = np.array([1.2, -0.4], dtype=np.float32)
    b2_np = np.float32(0.05)

    lr = 0.05
    final_loss = 0.0

    lib.mlp_loss_batch.restype = ctypes.c_float

    xs = FloatArray4(*xs_np)
    ys = FloatArray4(*ys_np)

    start = time.perf_counter()

    for step in range(steps):
        w1 = FloatArray2(*w1_np)
        b1 = FloatArray2(*b1_np)
        w2 = FloatArray2(*w2_np)

        dw1 = FloatArray2(0.0, 0.0)
        db1 = FloatArray2(0.0, 0.0)
        dw2 = FloatArray2(0.0, 0.0)
        db2 = ctypes.c_float(0.0)

        dxs = FloatArray4(0.0, 0.0, 0.0, 0.0)
        dys = FloatArray4(0.0, 0.0, 0.0, 0.0)

        final_loss = lib.mlp_loss_batch(
            xs,
            ys,
            w1,
            b1,
            w2,
            ctypes.c_float(b2_np),
        )

        
        lib.d_mlp_loss_batch(
            xs,
            dxs,
            ys,
            dys,
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
    xs_np = np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32)
    ys_np = np.sin(xs_np).astype(np.float32)

    steps = 1000

    single_lib = compile_loma(
        "mlp_forward.py",
        "_code/mlp_single_pipeline"
    )

    batch_lib = compile_loma(
        "mlp_batch.py",
        "_code/mlp_batch_pipeline"
    )

    single_loss, single_time = run_per_sample_training(
        single_lib,
        xs_np,
        ys_np,
        steps=steps,
    )

    batch_loss, batch_time = run_batch_training(
        batch_lib,
        xs_np,
        ys_np,
        steps=steps,
    )

    print("===== Batch pipeline benchmark =====")
    print("Steps:", steps)
    print("Batch size:", 4)
    print()
    print("Per-sample final loss:", single_loss)
    print("Per-sample time:", single_time)
    print()
    print("Batched final loss:", batch_loss)
    print("Batched time:", batch_time)
    print()
    print("Speedup per_sample_time / batch_time:", single_time / batch_time)


if __name__ == "__main__":
    main()