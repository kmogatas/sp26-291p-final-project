def mlp_loss(x : In[float], y_target : In[float], w1 : In[Array[float]], b1 : In[Array[float]], w2 : In[Array[float]], b2 : In[float]) -> float:
    z0 : float = w1[0] * x + b1[0]
    z1 : float = w1[1] * x + b1[1]

    h0 : float = 0.0
    h1 : float = 0.0

    if z0 > 0.0:
        h0 = z0
    else:
        h0 = 0.0

    if z1 > 0.0:
        h1 = z1
    else:
        h1 = 0.0

    y_pred : float = w2[0] * h0 + w2[1] * h1 + b2

    diff : float = y_pred - y_target
    loss : float = diff * diff

    return loss

d_mlp_loss = rev_diff(mlp_loss)