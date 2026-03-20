---
title: Frequentist Sample Size Estimation
date: 08/15/2025
categories:
- statistics
jupyter: python3
marimo-version: 0.21.1
---

```python {.marimo hide_code="true"}
import marimo as mo
```

In the previous post, I showed a Bayesian method of sample size estimation for A/B/n testing. This post goes over the more conventional frequentist method.

As before, here's the context. Say we're trying to test which variant of an email message generates the highest response rate. We consider $k$ different messages and send out $n$ emails for each message. After we wait for responses, we should be able to tell which message yielded the highest response rate as long as we set $n$ high enough. But we generally can't send out too many messages: say we're capped at $N$ total. How do we choose the highest $k$ that still allows us to confidently pick which message got the highest response rate?

## Frequentist Two-Arm Estimation

If we only have two messages (or 'arms' in the standard terminology), we might naively use a t-test to see if the observed difference in mean response rates $\hat{\delta}$ was statistically significant. Let $\hat{\sigma}$ be the pooled standard deviation. Then if the true difference in mean response rates $\delta$ is zero (the null hypothesis), $t = \frac{\sqrt{n}\hat{\delta}}{\sqrt{2}\hat{\sigma}}$ will be T distributed with $2n-2$ dof. To get a false positive rate of $\alpha$, we can reject the null hypothesis when $t > t_{1-\alpha}$ where $t_{1-\alpha}$ gives the $1-\alpha$ quantile of the Student's t distribution. Under the alternative hypothesis that $\delta = \delta_0 > 0$, $t$ will actually come from a non-central t distribution (ignoring the differences in arm variance). This means that the power of our test will be the survival function of a $2n-2$ degree non-central T distribution with noncentrality parameter $\delta_0 / \sigma$ evaluated at $t_{1-\alpha}$. We can find the number of samples $n$ that produces a desired power $\beta$ with a simple binary search. This calculation is implemented within the `statsmodels` library as follows:

```python {.marimo}
from statsmodels.stats import power
from scipy.optimize import brentq
from numpy.polynomial.hermite import hermgauss
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
from functools import partial
sns.set_theme()
```

```python {.marimo}
powers = [0.8, 0.85, 0.9]
```

```python {.marimo}
df = pd.concat([pd.DataFrame({
    'samples': [2 * power.tt_ind_solve_power(effect_size=delta, alpha=alpha, power=p, alternative='larger') for p in powers],
    'power': powers, 'delta': delta, 'alpha': alpha})
    for delta in [0.01, 0.015, 0.02]
    for alpha in [0.2, 0.15, 0.1]], ignore_index=True)
```

```python {.marimo}
def plot_t_test():
    for ax in sns.relplot(data=df, x='power', y='samples', hue='delta', col='alpha', kind='line').axes.ravel():
        ax.set_xticks(powers)
    return ax
plot_t_test()
```

This suggests we'd need a truly massive number of samples! Fortunately, things aren't as bad as they seem. We can use a Z-test instead.

Say response rates are $\theta_0$ and $\theta_1$ with difference $\delta$. The difference of observed arm means $\hat{\delta}$ will be approximately normal around $\delta$ with variance $\frac{\theta_0(1-\theta_0) + \theta_1(1-\theta_1)}{n}$. Let $\sigma_0^2 =2\theta_0(1-\theta_0)$ and  $\sigma_1^2 =\theta_0(1-\theta_0) + \theta_1(1-\theta_1)$.  If $\delta = 0$ and we reject when $\hat{\delta} > \frac{\sigma_0}{\sqrt{n}} z_{1 - \alpha}$, we'll have a false positive rate of $\alpha$. If $\delta = \delta_0$, on the other hand, then $\hat{\delta} - \delta_0$ will be normal with variance $\sigma_1^2/n$, so rejecting if $\hat{\delta} > \delta_0 + \frac{\sigma_1}{\sqrt{n}} z_{1 - \beta}$ would give us power $\beta$. We can choose $n$ so that these critical values are the same.

A little algebra tells us that $n = \frac{(\sigma_0 z_{1 - \alpha} - \sigma_1 z_{1 - \beta})^2}{\delta_0^2}$. Once again, this is already implemented in `statsmodels`.

```python {.marimo}
t = 0.04
```

```python {.marimo}
dfz = pd.concat([pd.DataFrame({
    'samples': [2 * power.normal_sample_size_one_tail(d, p, alpha,
                        std_null=np.sqrt(2 * t * (1-t)),
                        std_alternative=np.sqrt(t * (1-t) + (t+d)*(1 - (t+d))))
                for p in powers],
    'power': powers, 'delta': d, 'alpha': alpha})
    for d in [0.01, 0.015, 0.02]
    for alpha in [0.2, 0.15, 0.1]], ignore_index=True)
```

```python {.marimo}
def plot_z_test():
    for ax in sns.relplot(data=dfz, x='power', y='samples', hue='delta', col='alpha', kind='line').axes.ravel():
        ax.set_xticks(powers)
    return ax
plot_z_test()
```

## Tukey's Multi-Arm Estimation

With more than two arms, we end up testing more than one null hypothesis. One way to do this is to test a null hypothesis that arm $i$ has a higher mean than arm $j$ for every pair arms $i,j$. We'll reject this hypothesis when $\hat{\delta}_{i,j} > c$ for a specific value of $c$. Our family-wise false positive rate $1 - \alpha$ must now cover the probability that we mistakenly reject any of these tests. If $\hat{\delta}_{i,j} > c$ for any $i,j$, it will also be true that $\delta^\star > c$ where $\delta^\star$ is the largest observed difference in means between any pair of arms. We can show that $\delta^\star$ follows a known distribution.

In general, if $F$ is a distribution's cdf, the joint probability that each of $k$ independent draws from the distribution are between $a$ and $a+\delta$ is given by $(F(a + \delta) - F(a))^k$. This makes the density at $a$ given by $k f(a)(F(a + \delta) - F(a))^{k-1}$. We want to integrate over all $a$: $\int_{-\infty}^\infty k f(a)(F(a + \delta) - F(a))^{k-1}\, da$.

This distribution over the largest difference within $k$ samples is known as the *Studentized range distribution* when $F$ is the cdf of a Student's t distribution. This is built into `scipy`! And because a normal distribution is just a Student's t distribution with infinite degrees of freedom, this covers a largest difference between normal samples as well, which I'll refer to as a *range distribution*.

Say each of the per-arm differences are normal with approximately the same variance $\sigma^2$. We can reject the null hypothesis that all the inter-arm differences are zero when $\delta^\star > \frac{\sigma}{\sqrt{n}} q_{1-\alpha}$ to get a family-wise false positive rate of $\alpha$, where $q_{1-\alpha}$ gives the $1-\alpha$ quantile of the range distribution. If we also reject when $\delta^\star > \delta_0 +  \frac{\sigma}{\sqrt{n}} q_{1-\beta}$, we'd have power $\beta$ when the true maximum difference is $\delta_0$. Once again, we can solve for when these critical values are equal to find the necessary number of samples; unfortunately the algebra here isn't built into `statsmodels`, so we'll have to do it ourselves.

```python {.marimo}
def range_sample_size(k, theta=0.06, delta=0.015, alpha=0.90, beta=0.8):
    q0 = stats.studentized_range.ppf(1 - alpha, k, np.inf)
    q1 = stats.studentized_range.ppf(1 - beta, k, np.inf)
    theta1 = theta + delta
    return k * theta1 * (1 - theta1) * ((q0 - q1) / delta)**2
```

```python {.marimo}
ks = range(2, 5)
```

```python {.marimo}
dfk = pd.concat([pd.DataFrame({
    'samples': [range_sample_size(k, delta=delta, theta=theta, alpha=alpha) for k in ks],
    'k': ks, 'delta': delta, 'theta': theta, 'alpha': alpha})
    for delta in [0.01, 0.015, 0.02]
    for theta in [0.04, 0.05]
    for alpha in [0.2, 0.15, 0.1]], ignore_index=True)
```

```python {.marimo}
def plot_tukey():
    for ax in sns.relplot(data=dfk, x='k', y='samples', hue='delta', col='theta', row='alpha', kind='line').axes.ravel():
        ax.set_xticks(list(ks))
    return ax
plot_tukey()
```

## Dunnett's Many Arms

Another way of handling multiple arms is to use Dunnett's Test. Rather than comparing every pair of arms against each other, Dunnett's test compares each arm against the control arm. Specifically, we observe $k$ differences $\hat{\delta}_k \geq \hat{\delta}_{k-1}, \dotsc \hat{\delta}_1$ between each treatment arm and a control arm. We have $k$ null hypotheses $H_j = \{ \delta_i = 0 \}$.

As before, we'll reject $H_j$ when a function of $\delta_j$ is above a threshold. Specifically, we reject $H_j$ when $Y_j > c$ where $Y_j = \frac{\hat{\delta}_j}{\sqrt{2}\sigma_0}$, $\sigma_i = \sqrt{p_i(1-p_i)/n}$. To choose a $c$ to achieve a particular family-wise error rate, we need to do some algebra on the $Y_i$:

$$
\begin{align*}
Y_i &= \frac{A_i - B + \delta_i}{\sqrt{2}\sigma_0} \text{ where } A_i = \hat{p}_i - p_i``, ``B = \hat{p}_0 - p_0 \\
P(\text{max}_i\, Y_i < c) &= E \left[ \prod_i P(Y_i < c | B)\right] = E \left[ \prod_i P(A_i - B + \delta_i < \sqrt{2} c \sigma_0 | B) \right] \\
E \left[ \prod_i \Phi(\frac{\sqrt{2} c \sigma_0 + B - \delta_i}{\sigma_i}) \right] &= E \left[ \prod_i \Phi(\frac{\sqrt{2} c \sigma_0 + \sigma_0 Z - \delta_i}{\sigma_i}) \right]
\end{align*}
$$

 So if $p_i = p_0$, $E \left[ \prod_i \Phi(\frac{(\sqrt{2} c + Z)\sigma_0 - \delta_i}{\sigma_i}) \right] = E \left[ \Phi(\sqrt{2} c + Z)^k \right]$. To find $c$, we can solve for when this expresion equals $1 - \alpha$.

```python {.marimo}
nodes, weights = hermgauss(32)

def integrate_normal(f):
     return np.sum(weights * f(np.sqrt(2) * nodes)) / np.sqrt(np.pi)

def dunnett_c(k, α):
     def f(c, z):
         return stats.norm.cdf(np.sqrt(2) * c + z)**k
     return brentq(lambda c: integrate_normal(partial(f,c)) - (1 - α), 0, 5)
```

Given a threshold $c$, we now need to find the power of a Dunnet test. Using the same derivation is before, we find this is equal to

$$
E \left[ \prod_i \Phi(\frac{\sqrt{2} c \sigma_0 + \sigma_0Z - \delta_i}{\sigma_i}) \right] = E \left[ \Phi(\frac{\sqrt{2} c \sigma_0 + \sigma_0Z - \delta_k}{\sigma_k}) \Phi(\sqrt{2} c + Z)^{k-1} \right]
$$

```python {.marimo}
def dunnett_accept_prob(n, k, p0, pk, c):
    s0 = np.sqrt(p0 * (1 - p0) / n)
    sk = np.sqrt(pk * (1 - pk) / n)
    def f(z):
        best_arm = stats.norm.cdf((s0 * (np.sqrt(2) * c + z) - (pk - p0)) / sk)
        other_arms = stats.norm.cdf(np.sqrt(2) * c + z)**(k-1)
        return best_arm * other_arms
    return integrate_normal(f)
```

```python {.marimo}
def dunnett_sample_size(k, p0, p1, α, β):
    c = dunnett_c(k, α)
    return brentq(lambda n: dunnett_accept_prob(n, k, p0, p1, c) - (1 - β), 2.0, 100000.0)
```

```python {.marimo}
df_dunnet = pd.concat([pd.DataFrame({
    'samples': [dunnett_sample_size(k, theta, theta + delta, alpha, 0.8) for k in ks],
    'k': ks, 'delta': delta, 'theta': theta, 'alpha': alpha})
    for delta in [0.01, 0.015, 0.02]
    for theta in [0.04, 0.05]
    for alpha in [0.2, 0.15, 0.1]], ignore_index=True)
```

```python {.marimo}
def plot_dunnet():
    for ax in sns.relplot(data=df_dunnet, x='k', y='samples', hue='delta', col='theta', row='alpha', kind='line').axes.ravel():
        ax.set_xticks(list(ks))
    return ax
plot_dunnet()
```