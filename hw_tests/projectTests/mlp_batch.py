def mlp_loss_batch(xs : In[Array[float]], ys : In[Array[float]], w1 : In[Array[float]], b1 : In[Array[float]], w2 : In[Array[float]], b2 : In[float]) -> float:
    total_loss : float = 0.0
    i : int = 0

    x : float = 0.0
    y_target : float = 0.0

    z0 : float = 0.0
    z1 : float = 0.0

    h0 : float = 0.0
    h1 : float = 0.0

    y_pred : float = 0.0
    diff : float = 0.0

    while (i < 4, max_iter := 4):
        x = xs[i]
        y_target = ys[i]

        z0 = w1[0] * x + b1[0]
        z1 = w1[1] * x + b1[1]

        h0 = log(1.0 + exp(z0))
        h1 = log(1.0 + exp(z1))

        y_pred = w2[0] * h0 + w2[1] * h1 + b2

        diff = y_pred - y_target
        total_loss = total_loss + diff * diff

        i = i + 1

    return total_loss / 4.0

d_mlp_loss_batch = rev_diff(mlp_loss_batch)