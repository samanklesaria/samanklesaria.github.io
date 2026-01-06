---
categories: ["math"]
date: 2024-02-13
title: Krylov Methods
excerpt: >
  The $i$th *Krylov subspace* $\mathcal{K}_i$ for a symmetric matrix $A$
  is the subspace spanned by repeatedly multiplying $A$ by an initial vector $b$.
  This most demonstrates algorithms that take advantage of the special structure of this subspace.
---

The $i$th *Krylov subspace* $\mathcal{K}_i$ for a symmetric matrix $A$ starting from vector $b$ is the subspace spanned by the vectors $b, Ab, A^2b, \dotsc A^{i-1}b$. Algorithms that use these subspaces are called Krylov subspace methods. To see why this subspace can be useful, it helps to see some examples.


### The Lanczos Algorithm

The Lanczos algorithm finds an orthonormal matrix $V$ and tridiagonal matrix $T$ so that $A = VTV^T$. The columns of $V$ are chosen from successive Krylov subspaces. $v_1$ is chosen to be $b/ \|b\|$. Afterwards, $v_i$ is just $Av_{i-1}$, but normalized and without any components parallel to the previous columns. Specifically
$$
\begin{align*}
w_{i+1}' &= Av_i \\
w_{i+1} &= w_{i+1}' - \sum_{k=1}^i \langle v_k, w_{i+1}'\rangle v_k \\
v_{i+1} &= w_{i+1} / \|w_{i+1}\|
\end{align*}
$$
Note that this means $Av_k$ is a linear combination of $v_1$ through $v_{k+1}$, so
$$
\begin{align*}
\langle v_k , w_{i+1}' \rangle &= \langle v_k , A v_i \rangle \\
&= \langle Av_k, v_i \rangle \\
&= \left \langle \sum_{j=1}^{k+1} c_j v_j , v_i \right\rangle
\end{align*}
$$
for some constants $c_j$. As the $v_i$ are chosen to be orthogonal, the inner product will be zero whenever $k+1<i$. This means that we only need to orthogonalize each $w_{i+1}’$ with respect to $v_i$ and $v_{i-1}$; the rest will already be orthogonal. This property shows that $T = V^TAV$ (with entry $(i,j)$ of the form $v_i^TAv_j$) must be  tridiagonal as required.

Finally, note that
$$
\begin{align*}
\langle v_{i-1} , w_{i+1}' \rangle &= \langle Av_{i-1}, v_i \rangle \\
&= \langle w_i', v_i \rangle \\
&= \langle w_i + c_1 v_{i-1} + c_2 v_{i-2}, v_i \rangle \\
&= \langle w_i , v_i \rangle \\
&= \langle w_i , w_i \rangle / \|w_i\| = \|w_i\|
\end{align*}
$$


We already calculated this quantity when normalizing to get $v_i$, so this saves us an extra multiplication at each step.





### The Conjugate Gradients Algorithm

The conjugate gradients algorithm computes $A^{-1}b$ using a similar strategy. While the Lanczos algorithm makes $v_i$ orthogonal to the previous $v_j$, the conjugate gradients algorithm makes it $A$-orthogonal. We say that $v_i$ and $v_j$ are $A$-orthogonal when $v_i^TAv_j = 0$. I will also write $v_i^TAv_j$ as $\langle v_i, v_j \rangle_A$.

Say there was a matrix where $u = \sum_t c_t v_t$ for $A$-orthogonal $v_t$. Then $\langle v_i,u \rangle_A = c_i \langle v_i, v_i \rangle_A$, so $c_i = \frac{\langle v_i u \rangle_A}{\langle v_i, v_i \rangle_A}$. This means we can make $u$ $A$-orthogonal to $v_i$ by subtracting $\frac{\langle v_i u \rangle_A}{\langle v_i, v_i \rangle_A} v_i$.

This lets conjugate gradients compute the vectors $v_i$ as follows:
$$
\begin{align*}
w_{i+1}' &= Av_i \\
w_{i+1} &= w_{i+1}' - \sum_{k=1}^i \frac{\langle v_k, w_{i+1}'\rangle_A}{\langle v_k, v_k\rangle_A}v_k \\
v_{i+1} &= w_{i+1}/\|w_{i+1}\|
\end{align*}
$$
This is exactly the same as the computation of the $v_i$ in the Lanczos algorithm except that the inner products use the $A$ metric.

Once we have an $A$-orthogonal basis like this, we can find an $x$ to minimize $\|x - A^{-1}b\|_A$ (which is equivalent to finding $A^{-1}b$ if it exists). This is the same as minimizing $x^TAx -2x^Tb$. Expressing $x$ as $\sum_i \alpha_i v_i$ and using A-orthogonality gives
$$
\sum_i (\alpha_i)^2 v_i^TAv_i - 2\alpha_iv_i^Tb
$$
The derivative is zero when $\alpha_i = \frac{v_i^Tb}{v_i^TAv_i}$. This lets us solve $A^{-1}b$ in a linear number of matrix vector multiplications. If  multiplication by $A$ is $O(n^2)$, then this isn’t any better than $O(n^3)$ Gaussian elimination. But it’s often the case that multiplication by $A$ is much faster than that, especially if it’s sparse or Toeplitz or comes from a Kronecker product.

