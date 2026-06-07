@simd
def mlp_train_batch(
    x : In[Array[float]],
    y_target : In[Array[float]],
    w1 : In[Array[float]],
    b1 : In[Array[float]],
    w2 : In[Array[float]],
    b2 : In[Array[float]],
    loss_out : Out[Array[float]],
    dw1_out : Out[Array[float]],
    db1_out : Out[Array[float]],
    dw2_out : Out[Array[float]],
    db2_out : Out[Array[float]]):
    i : int = thread_id()

    x_i : float = x[i]
    y_i : float = y_target[i]

    z0 : float = w1[0] * x_i + b1[0]
    z1 : float = w1[1] * x_i + b1[1]

    h0 : float = log(1.0 + exp(z0))
    h1 : float = log(1.0 + exp(z1))

    y_pred : float = w2[0] * h0 + w2[1] * h1 + b2[0]
    diff : float = y_pred - y_i
    loss : float = diff * diff

    loss_out[i] = loss

    dloss : float = 2.0 * diff
    dw2_out[2 * i + 0] = dloss * h0
    dw2_out[2 * i + 1] = dloss * h1
    db2_out[i] = dloss

    dh0 : float = dloss * w2[0]
    dh1 : float = dloss * w2[1]
    sigmoid0 : float = 1.0 / (1.0 + exp(-z0))
    sigmoid1 : float = 1.0 / (1.0 + exp(-z1))

    dz0 : float = dh0 * sigmoid0
    dz1 : float = dh1 * sigmoid1

    dw1_out[2 * i + 0] = dz0 * x_i
    dw1_out[2 * i + 1] = dz1 * x_i
    db1_out[2 * i + 0] = dz0
    db1_out[2 * i + 1] = dz1

@simd
def reduce_gradients(
    dw1_in : In[Array[float]],
    db1_in : In[Array[float]],
    dw2_in : In[Array[float]],
    db2_in : In[Array[float]],
    loss_in : In[Array[float]],
    batch_size : In[int],
    grad_w1_out : Out[Array[float]],
    grad_b1_out : Out[Array[float]],
    grad_w2_out : Out[Array[float]],
    grad_b2_out : Out[Array[float]],
    loss_out : Out[Array[float]]):
    tid : int = thread_id()
    
    sum_w1_0 : float = 0.0
    sum_w1_1 : float = 0.0
    sum_b1_0 : float = 0.0
    sum_b1_1 : float = 0.0
    sum_w2_0 : float = 0.0
    sum_w2_1 : float = 0.0
    sum_b2 : float = 0.0
    sum_loss : float = 0.0
    
    idx : int = tid
    while (idx < batch_size, max_iter := 4096):
        sum_w1_0 = sum_w1_0 + dw1_in[2 * idx + 0]
        sum_w1_1 = sum_w1_1 + dw1_in[2 * idx + 1]
        sum_b1_0 = sum_b1_0 + db1_in[2 * idx + 0]
        sum_b1_1 = sum_b1_1 + db1_in[2 * idx + 1]
        sum_w2_0 = sum_w2_0 + dw2_in[2 * idx + 0]
        sum_w2_1 = sum_w2_1 + dw2_in[2 * idx + 1]
        sum_b2 = sum_b2 + db2_in[idx]
        sum_loss = sum_loss + loss_in[idx]
        idx = idx + 64
    
    atomic_add(grad_w1_out[0], sum_w1_0)
    atomic_add(grad_w1_out[1], sum_w1_1)
    atomic_add(grad_b1_out[0], sum_b1_0)
    atomic_add(grad_b1_out[1], sum_b1_1)
    atomic_add(grad_w2_out[0], sum_w2_0)
    atomic_add(grad_w2_out[1], sum_w2_1)
    atomic_add(grad_b2_out[0], sum_b2)
    atomic_add(loss_out[0], sum_loss)

@simd
def clear_accumulators(
    grad_w1 : Out[Array[float]],
    grad_b1 : Out[Array[float]],
    grad_w2 : Out[Array[float]],
    grad_b2 : Out[Array[float]],
    loss_sum : Out[Array[float]]):
    i : int = thread_id()
    if i == 0:
        grad_w1[0] = 0.0
        grad_w1[1] = 0.0
        grad_b1[0] = 0.0
        grad_b1[1] = 0.0
        grad_w2[0] = 0.0
        grad_w2[1] = 0.0
        grad_b2[0] = 0.0
        loss_sum[0] = 0.0

@simd
def mlp_train_batch_accumulate(
    x : In[Array[float]],
    y_target : In[Array[float]],
    w1 : In[Array[float]],
    b1 : In[Array[float]],
    w2 : In[Array[float]],
    b2 : In[Array[float]],
    grad_w1 : Out[Array[float]],
    grad_b1 : Out[Array[float]],
    grad_w2 : Out[Array[float]],
    grad_b2 : Out[Array[float]],
    loss_sum : Out[Array[float]]):
    i : int = thread_id()

    x_i : float = x[i]
    y_i : float = y_target[i]

    z0 : float = w1[0] * x_i + b1[0]
    z1 : float = w1[1] * x_i + b1[1]

    h0 : float = log(1.0 + exp(z0))
    h1 : float = log(1.0 + exp(z1))

    y_pred : float = w2[0] * h0 + w2[1] * h1 + b2[0]
    diff : float = y_pred - y_i
    loss : float = diff * diff

    dloss : float = 2.0 * diff
    dh0 : float = dloss * w2[0]
    dh1 : float = dloss * w2[1]
    sigmoid0 : float = 1.0 / (1.0 + exp(-z0))
    sigmoid1 : float = 1.0 / (1.0 + exp(-z1))

    dz0 : float = dh0 * sigmoid0
    dz1 : float = dh1 * sigmoid1

    atomic_add(grad_w1[0], dz0 * x_i)
    atomic_add(grad_w1[1], dz1 * x_i)
    atomic_add(grad_b1[0], dz0)
    atomic_add(grad_b1[1], dz1)
    atomic_add(grad_w2[0], dloss * h0)
    atomic_add(grad_w2[1], dloss * h1)
    atomic_add(grad_b2[0], dloss)
    atomic_add(loss_sum[0], loss)

@simd
def update_params_gpu(
    w1 : Out[Array[float]],
    b1 : Out[Array[float]],
    w2 : Out[Array[float]],
    b2 : Out[Array[float]],
    grad_w1 : In[Array[float]],
    grad_b1 : In[Array[float]],
    grad_w2 : In[Array[float]],
    grad_b2 : In[Array[float]],
    lr : In[float],
    batch_size : In[int]):
    i : int = thread_id()
    scale : float = lr / batch_size
    if i == 0:
        w1[0] = w1[0] - scale * grad_w1[0]
        w1[1] = w1[1] - scale * grad_w1[1]
        b1[0] = b1[0] - scale * grad_b1[0]
        b1[1] = b1[1] - scale * grad_b1[1]
        w2[0] = w2[0] - scale * grad_w2[0]
        w2[1] = w2[1] - scale * grad_w2[1]
        b2[0] = b2[0] - scale * grad_b2[0]

@simd
def update_params_and_clear_gpu(
    w1 : Out[Array[float]],
    b1 : Out[Array[float]],
    w2 : Out[Array[float]],
    b2 : Out[Array[float]],
    grad_w1 : Out[Array[float]],
    grad_b1 : Out[Array[float]],
    grad_w2 : Out[Array[float]],
    grad_b2 : Out[Array[float]],
    loss_sum : Out[Array[float]],
    loss_mean : Out[Array[float]],
    lr : In[float],
    batch_size : In[int]):
    i : int = thread_id()
    scale : float = lr / batch_size
    mean_scale : float = 1.0 / batch_size
    if i == 0:
        loss_mean[0] = loss_sum[0] * mean_scale

        w1[0] = w1[0] - scale * grad_w1[0]
        w1[1] = w1[1] - scale * grad_w1[1]
        b1[0] = b1[0] - scale * grad_b1[0]
        b1[1] = b1[1] - scale * grad_b1[1]
        w2[0] = w2[0] - scale * grad_w2[0]
        w2[1] = w2[1] - scale * grad_w2[1]
        b2[0] = b2[0] - scale * grad_b2[0]

        grad_w1[0] = 0.0
        grad_w1[1] = 0.0
        grad_b1[0] = 0.0
        grad_b1[1] = 0.0
        grad_w2[0] = 0.0
        grad_w2[1] = 0.0
        grad_b2[0] = 0.0
        loss_sum[0] = 0.0
