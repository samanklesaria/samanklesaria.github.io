---
title: Generative ODE Models are VAEs
categories: ["machine_learning"]
date: 2023-08-15
---

Generative image models based on ordinary differential equations can be seen as forms of variational auto-encoders with a partially deterministic inference network.
$\newcommand{\coloneqq}{\mathrel{\vcenter{:}}=}$

# Variational Auto-encoders

Given a dataset of samples $y$ from an unknown distribution $\pi_1$, generative models attempt to find parameters $\theta$ for a fixed family  that maximizes the likelihood $p_\theta(y)$. When the family implicitly marginalizes over a latent variable $z$ so that $\log p_\theta(y) = \log E_{z \sim p} p_\theta(y \vert  z)$, maximum likelihood estimation becomes intractable in general. Variational auto-encoders address this problem by maximizing a lower bound on the likelihood:
$$
\begin{align*}
\log p_\theta(y) &= \log E_{z \sim p} p_\theta(y \vert  z) \\
&=\log E_{q(z \vert  y)} \frac{p(y\vert z) p(z)}{q(z\vert y)} &\text{ (importance sampling)}\\
& \geq E_{q(z \vert  y)} \log \frac{p(y\vert z) p(z)}{q(z)} &\text{ (Jensen's inequality)}
\end{align*}
$$

This quantity $\mathcal{L} \coloneqq E_{q(z \vert  y)} \log p(y,z) - \log q(z)$ is known as the *evidence lower bound* or ELBO. The distribution $q(z\vert y)$ used for importance sampling is usually parameterized by a neural net and is known as the *inference network*. In what follows, we will implicitly assume that $q$ is a function of neural net parameters $\phi$.



# From Variational Auto-encoders to Rectified Flows

Assume that the model's prior over $z$ factors as $p(z) = p(z_0)\prod_t p(z_{t+\Delta} \vert  z_t)$, where $t$ ranges from $0$ to $1$ in increments of $\Delta$ and $z_1$ is observed (taking the place of $y$ in the discussion so far). Instead of fixing the model $p$ and learning parameters for a distribution $q$ to approximate its posterior, with diffusion-based approaches, we do the opposite.. Fix the posterior approximation $q(z_t \vert  z_1, z_0)$ to be a Delta distribution at $tz_1 + (1 - t)z_0$ (linearly interpolating between observations). Let $p(z_{t+\Delta} \vert  z_t)$ be distributed as $\mathcal{N}(z_t + \Delta f_\theta(z_t, t), \sigma^2/\Delta^2)$, where $f_\theta$ is a neural network parameterized by $\theta$.
Under this linear interpolation scheme, $z_{t + \Delta} - z_t = \Delta (z_1 - z_0)$. This means that
$$
\begin{align*}
    &E\left[\log p((z_{t+\Delta} \vert  z_t)\right] \\
    &= E\left[-\frac{1}{2} \frac{(z_{t + \Delta} - [z_t + \Delta f_\theta (z_t, t)])^2}{\sigma^2 \Delta^2}\right] \\
    &= E\left[\frac{-1}{2\sigma^2 \Delta} ((z_1 - z_0) - f_\theta (z_t, t))^2\right]
\end{align*}
$$

We can plug this simplification into the full ELBO, noting that the sum over time can be written as an expectation.

$$
E_{z_0} \left[ \log p(z_0) - \log q(z_0\vert z_1) + \sum_t \log p(z_{t+\Delta} \vert  z_t) \right] = \\
-KL[q(z_0\vert z_1) \vert p(z_0)] + \frac{-1}{2\sigma^2} E_{t, z_0} \left[ ((z_1 - z_0) - f_\theta (z_t, t))^2 \right]
$$

It breaks into two parts: a KL divergence keeping $q(z_0\vert z_1)$ close to $p(z_0)$, and a stochastic squared loss in which both time and latent code are sampled. As $\Delta \to 0$, the distribution over times approaches uniform. This is precisely the loss for the *Rectified Flow* model for the case of a fixed $q(z_0 \vert  z_1)$.  Originally, the authors interpreted this loss as a way to learn an ODE $\frac{\partial z}{\partial t} = f(z, t)$ which transports samples $z_0$ to observations $z_1$. By casting the loss in the language of variational auto-encoders, however, we can learn a posterior distribution for $z_0$ at the same time.
