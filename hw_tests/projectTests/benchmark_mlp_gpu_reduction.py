import os
import sys
import time
import ctypes
import numpy as np
import torch
import torch_directml
import compiler
import slang_utils
import slangpy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


print("PyTorch device:", "cuda" if torch.cuda.is_available() else "CPU only")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

slang_device = slang_utils.create_slang_device()
print("Slang device:", slang_device)

def train_loma_slang_with_gpu_reduction(batch_size=1024, steps=100, lr=0.05):

    slang_device = slang_utils.create_slang_device()
    
    with open(os.path.join(os.path.dirname(__file__), "mlp_forward_slang_gpu_reduce.py")) as f:
        module, kernels = compiler.compile(
            f.read(),
            target="slang",
            slang_device=slang_device,
        )

    x_np = np.linspace(-3.14, 3.14, batch_size, dtype=np.float32)
    y_np = np.sin(x_np).astype(np.float32)

    w1_np = np.array([0.5, -0.8], dtype=np.float32)
    b1_np = np.array([0.1, 0.2], dtype=np.float32)
    w2_np = np.array([1.2, -0.4], dtype=np.float32)
    b2_np = np.array([0.05], dtype=np.float32)

    x_buf = slangpy.Tensor.from_numpy(device=slang_device, ndarray=x_np)
    y_buf = slangpy.Tensor.from_numpy(device=slang_device, ndarray=y_np)
    w1_buf = slangpy.Tensor.from_numpy(device=slang_device, ndarray=w1_np)
    b1_buf = slangpy.Tensor.from_numpy(device=slang_device, ndarray=b1_np)
    w2_buf = slangpy.Tensor.from_numpy(device=slang_device, ndarray=w2_np)
    b2_buf = slangpy.Tensor.from_numpy(device=slang_device, ndarray=b2_np)

    grad_w1_buf = slangpy.Tensor.zeros(device=slang_device, dtype=float, shape=(2,))
    grad_b1_buf = slangpy.Tensor.zeros(device=slang_device, dtype=float, shape=(2,))
    grad_w2_buf = slangpy.Tensor.zeros(device=slang_device, dtype=float, shape=(2,))
    grad_b2_buf = slangpy.Tensor.zeros(device=slang_device, dtype=float, shape=(1,))
    loss_sum_buf = slangpy.Tensor.zeros(device=slang_device, dtype=float, shape=(1,))
    loss_mean_buf = slangpy.Tensor.zeros(device=slang_device, dtype=float, shape=(1,))

    start = time.perf_counter()

    for step in range(steps):
        kernels['mlp_train_batch_accumulate'].dispatch(
            thread_count=[batch_size, 1, 1],
            _total_threads=batch_size,
            x=x_buf.storage,
            y_target=y_buf.storage,
            w1=w1_buf.storage,
            b1=b1_buf.storage,
            w2=w2_buf.storage,
            b2=b2_buf.storage,
            grad_w1=grad_w1_buf.storage,
            grad_b1=grad_b1_buf.storage,
            grad_w2=grad_w2_buf.storage,
            grad_b2=grad_b2_buf.storage,
            loss_sum=loss_sum_buf.storage,
        )

        kernels['update_params_and_clear_gpu'].dispatch(
            thread_count=[1, 1, 1],
            _total_threads=1,
            w1=w1_buf.storage,
            b1=b1_buf.storage,
            w2=w2_buf.storage,
            b2=b2_buf.storage,
            grad_w1=grad_w1_buf.storage,
            grad_b1=grad_b1_buf.storage,
            grad_w2=grad_w2_buf.storage,
            grad_b2=grad_b2_buf.storage,
            loss_sum=loss_sum_buf.storage,
            loss_mean=loss_mean_buf.storage,
            lr=float(lr),
            batch_size=int(batch_size),
        )

    final_loss = float(loss_mean_buf.to_numpy().astype(np.float32)[0])
    end = time.perf_counter()
    return final_loss, end - start


def train_loma_slang_with_staged_reduction(batch_size=1024, steps=100, lr=0.05):
    slang_device = slang_utils.create_slang_device()

    with open(os.path.join(os.path.dirname(__file__), "mlp_forward_slang_gpu_reduce.py")) as f:
        module, kernels = compiler.compile(
            f.read(),
            target="slang",
            slang_device=slang_device,
        )

    x_np = np.linspace(-3.14, 3.14, batch_size, dtype=np.float32)
    y_np = np.sin(x_np).astype(np.float32)

    w1_np = np.array([0.5, -0.8], dtype=np.float32)
    b1_np = np.array([0.1, 0.2], dtype=np.float32)
    w2_np = np.array([1.2, -0.4], dtype=np.float32)
    b2_np = np.array([0.05], dtype=np.float32)

    x_buf = slangpy.Tensor.from_numpy(device=slang_device, ndarray=x_np)
    y_buf = slangpy.Tensor.from_numpy(device=slang_device, ndarray=y_np)
    w1_buf = slangpy.Tensor.from_numpy(device=slang_device, ndarray=w1_np)
    b1_buf = slangpy.Tensor.from_numpy(device=slang_device, ndarray=b1_np)
    w2_buf = slangpy.Tensor.from_numpy(device=slang_device, ndarray=w2_np)
    b2_buf = slangpy.Tensor.from_numpy(device=slang_device, ndarray=b2_np)

    dw1_buf = slangpy.Tensor.empty(device=slang_device, dtype=float, shape=(batch_size * 2,))
    db1_buf = slangpy.Tensor.empty(device=slang_device, dtype=float, shape=(batch_size * 2,))
    dw2_buf = slangpy.Tensor.empty(device=slang_device, dtype=float, shape=(batch_size * 2,))
    db2_buf = slangpy.Tensor.empty(device=slang_device, dtype=float, shape=(batch_size,))
    loss_buf = slangpy.Tensor.empty(device=slang_device, dtype=float, shape=(batch_size,))

    grad_w1_buf = slangpy.Tensor.zeros(device=slang_device, dtype=float, shape=(2,))
    grad_b1_buf = slangpy.Tensor.zeros(device=slang_device, dtype=float, shape=(2,))
    grad_w2_buf = slangpy.Tensor.zeros(device=slang_device, dtype=float, shape=(2,))
    grad_b2_buf = slangpy.Tensor.zeros(device=slang_device, dtype=float, shape=(1,))
    loss_sum_buf = slangpy.Tensor.zeros(device=slang_device, dtype=float, shape=(1,))
    loss_mean_buf = slangpy.Tensor.zeros(device=slang_device, dtype=float, shape=(1,))

    start = time.perf_counter()

    for step in range(steps):
        kernels['mlp_train_batch'].dispatch(
            thread_count=[batch_size, 1, 1],
            _total_threads=batch_size,
            x=x_buf.storage,
            y_target=y_buf.storage,
            w1=w1_buf.storage,
            b1=b1_buf.storage,
            w2=w2_buf.storage,
            b2=b2_buf.storage,
            loss_out=loss_buf.storage,
            dw1_out=dw1_buf.storage,
            db1_out=db1_buf.storage,
            dw2_out=dw2_buf.storage,
            db2_out=db2_buf.storage,
        )

        kernels['reduce_gradients'].dispatch(
            thread_count=[64, 1, 1],
            _total_threads=64,
            dw1_in=dw1_buf.storage,
            db1_in=db1_buf.storage,
            dw2_in=dw2_buf.storage,
            db2_in=db2_buf.storage,
            loss_in=loss_buf.storage,
            batch_size=batch_size,
            grad_w1_out=grad_w1_buf.storage,
            grad_b1_out=grad_b1_buf.storage,
            grad_w2_out=grad_w2_buf.storage,
            grad_b2_out=grad_b2_buf.storage,
            loss_out=loss_sum_buf.storage,
        )

        kernels['update_params_and_clear_gpu'].dispatch(
            thread_count=[1, 1, 1],
            _total_threads=1,
            w1=w1_buf.storage,
            b1=b1_buf.storage,
            w2=w2_buf.storage,
            b2=b2_buf.storage,
            grad_w1=grad_w1_buf.storage,
            grad_b1=grad_b1_buf.storage,
            grad_w2=grad_w2_buf.storage,
            grad_b2=grad_b2_buf.storage,
            loss_sum=loss_sum_buf.storage,
            loss_mean=loss_mean_buf.storage,
            lr=float(lr),
            batch_size=int(batch_size),
        )

    final_loss = float(loss_mean_buf.to_numpy().astype(np.float32)[0])
    end = time.perf_counter()
    return final_loss, end - start


def train_pytorch_softplus(batch_size=1024, steps=100, lr=0.05):

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu') 

    x = torch.linspace(-3.14, 3.14, batch_size, device=device, dtype=torch.float32)
    y_target = torch.sin(x)

    w1 = torch.tensor([0.5, -0.8], dtype=torch.float32, device=device, requires_grad=True)
    b1 = torch.tensor([0.1, 0.2], dtype=torch.float32, device=device, requires_grad=True)
    w2 = torch.tensor([1.2, -0.4], dtype=torch.float32, device=device, requires_grad=True)
    b2 = torch.tensor(0.05, dtype=torch.float32, device=device, requires_grad=True)

    final_loss = None
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(steps):
        z = x.view(batch_size, 1) * w1.view(1, 2) + b1.view(1, 2)
        h = torch.log1p(torch.exp(z))
        y_pred = torch.sum(w2.view(1, 2) * h, dim=1) + b2
        loss = torch.mean((y_pred - y_target) ** 2)

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

        final_loss = float(loss.item())

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.perf_counter()
    return final_loss, end - start

def train_pytorch_softplus_gpu(batch_size=1024, steps=100, lr=0.05):

    device = torch_directml.device()

    x = torch.linspace(-3.14, 3.14, batch_size, device=device, dtype=torch.float32)
    y_target = torch.sin(x)

    w1 = torch.tensor([0.5, -0.8], dtype=torch.float32, device=device, requires_grad=True)
    b1 = torch.tensor([0.1, 0.2], dtype=torch.float32, device=device, requires_grad=True)
    w2 = torch.tensor([1.2, -0.4], dtype=torch.float32, device=device, requires_grad=True)
    b2 = torch.tensor(0.05, dtype=torch.float32, device=device, requires_grad=True)

    for _ in range(5):
        z = x.view(batch_size, 1) * w1.view(1, 2) + b1.view(1, 2)
        h = torch.log1p(torch.exp(z))
        y_pred = torch.sum(w2.view(1, 2) * h, dim=1) + b2
        loss = torch.mean((y_pred - y_target) ** 2)
        loss.backward()
        with torch.no_grad():
            for p in [w1, b1, w2, b2]:
                p -= lr * p.grad
                p.grad.zero_()

    w1 = torch.tensor([0.5, -0.8], dtype=torch.float32, device=device, requires_grad=True)
    b1 = torch.tensor([0.1, 0.2], dtype=torch.float32, device=device, requires_grad=True)
    w2 = torch.tensor([1.2, -0.4], dtype=torch.float32, device=device, requires_grad=True)
    b2 = torch.tensor(0.05, dtype=torch.float32, device=device, requires_grad=True)

    _ = w1.sum().item()
    start = time.perf_counter()

    final_loss = None
    for _ in range(steps):
        z = x.view(batch_size, 1) * w1.view(1, 2) + b1.view(1, 2)
        h = torch.log1p(torch.exp(z))
        y_pred = torch.sum(w2.view(1, 2) * h, dim=1) + b2
        loss = torch.mean((y_pred - y_target) ** 2)
        loss.backward()
        with torch.no_grad():
            for p in [w1, b1, w2, b2]:
                p -= lr * p.grad
                p.grad.zero_()
        final_loss = float(loss.item())

    end = time.perf_counter()
    return final_loss, end - start

def train_loma_c_cpu(batch_size=1024, steps=100, lr=0.05):
    with open(os.path.join(os.path.dirname(__file__), "mlp_forward_slang_gpu_reduce.py")) as f:
        structs, lib = compiler.compile(
            f.read(),
            target="c",
            output_filename="_loma_cpu_mlp",
        )

    x_np = np.linspace(-3.14, 3.14, batch_size, dtype=np.float32)
    y_np = np.sin(x_np).astype(np.float32)

    w1_np = np.array([0.5, -0.8], dtype=np.float32)
    b1_np = np.array([0.1,  0.2], dtype=np.float32)
    w2_np = np.array([1.2, -0.4], dtype=np.float32)
    b2_np = np.array([0.05],      dtype=np.float32)

    grad_w1 = np.zeros(2, dtype=np.float32)
    grad_b1 = np.zeros(2, dtype=np.float32)
    grad_w2 = np.zeros(2, dtype=np.float32)
    grad_b2 = np.zeros(1, dtype=np.float32)
    loss_sum = np.zeros(1, dtype=np.float32)
    loss_mean = np.zeros(1, dtype=np.float32)

    start = time.perf_counter()

    for _ in range(steps):
        lib.mlp_train_batch_accumulate(
            x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w1_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b1_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w2_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b2_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            grad_w1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            grad_b1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            grad_w2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            grad_b2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            loss_sum.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(batch_size),
        )

        scale = lr / batch_size
        loss_mean[0] = loss_sum[0] / batch_size
        w1_np -= scale * grad_w1
        b1_np -= scale * grad_b1
        w2_np -= scale * grad_w2
        b2_np -= scale * grad_b2

        grad_w1[:] = 0; grad_b1[:] = 0
        grad_w2[:] = 0; grad_b2[:] = 0
        loss_sum[:] = 0

    end = time.perf_counter()
    return float(loss_mean[0]), end - start

def main():
    batch_size = 65536 
    steps = 100

    print('===== MLP Benchmark: Loma GPU vs PyTorch GPU vs CPU =====')
    print(f'batch_size: {batch_size}, steps: {steps}')
    if torch is not None and torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print()

    atomic_loss, atomic_time   = train_loma_slang_with_gpu_reduction(batch_size, steps)
    staged_loss, staged_time   = train_loma_slang_with_staged_reduction(batch_size, steps)
    numpy_loss,  numpy_time    = train_loma_c_cpu(batch_size, steps)
    pytorch_cpu_loss, pytorch_cpu_time = train_pytorch_softplus(batch_size, steps)
    pytorch_gpu_loss, pytorch_gpu_time = train_pytorch_softplus_gpu(batch_size, steps)

    print('\n===== Results =====')
    results = [
        ('Loma GPU atomic',    atomic_loss,      atomic_time),
        ('Loma GPU staged',    staged_loss,       staged_time),
        ('Loma CPU',          numpy_loss,        numpy_time),
        ('PyTorch CPU',        pytorch_cpu_loss,  pytorch_cpu_time),
        ('PyTorch GPU',        pytorch_gpu_loss,  pytorch_gpu_time),
    ]

    baseline_time = pytorch_cpu_time
    for name, loss, t in results:
        if t is None:
            print(f'  {name:<25} SKIPPED')
            continue
        ratio = t / baseline_time
        print(f'  {name:<25} loss={loss:.6f}  time={t:.4f}s  ({ratio:.2f}x vs PyTorch CPU)')


if __name__ == '__main__':
    main()
