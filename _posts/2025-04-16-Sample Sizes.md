---
title: Bayesian and Frequentist Sample Size Estimation
category: math
classes: wide
excerpt: >
  This post reviews two approaches to sample size estimation in A/B/n testing: one Frequentist, and one Bayesian. Say we're trying to test which variant of an email message generates the highest response rate. We consider $k$ different messages and send out $n$ emails for each message. After we wait for responses, we should be able to tell which message yielded the highest response rate as long as we set $n$ high enough. But we generally can't send out too many messages: say we're capped at $N$ total. How do we choose the highest $k$ that still allows us to confidently pick which message got the highest response rate?
---

# Sample Size Estimation

This post reviews two approaches to sample size estimation in A/B/n testing: one Frequentist, and one Bayesian. Say we're trying to test which variant of an email message generates the highest response rate. We consider $k$ different messages and send out $n$ emails for each message. After we wait for responses, we should be able to tell which message yielded the highest response rate as long as we set $n$ high enough. But we generally can't send out too many messages: say we're capped at $N$ total. How do we choose the highest $k$ that still allows us to confidently pick which message got the highest response rate?

## Frequentist Two-Arm Estimation

If we only have two messages (or 'arms' in the standard terminology), we might naively use a t-test to see if the observed difference in mean response rates $\hat{\delta}$ was statistically significant. Let $\hat{\sigma}$ be the pooled standard deviation. Then if the true difference in mean response rates $\delta$ is zero (the null hypothesis), $t = \frac{\sqrt{n}\hat{\delta}}{\sqrt{2}\hat{\sigma}}$ will be T distributed with $2n-2$ dof. To get a false positive rate of $\alpha$, we can reject the null hypothesis when $t > t_{1-\alpha}$ where $t_{1-\alpha}$ gives the $1-\alpha$ quantile of the Student's t distribution. Under the alternative hypothesis that $\delta = \delta_0 > 0$, $t$ will actually come from a non-central t distribution (ignoring the differences in arm variance). We can see this because $\sqrt{n}(\hat{\delta} - \delta_0)/\sqrt{2}\sigma$ comes from a standard normal distribution, so $\sqrt{n}\hat{\delta}/\sqrt{2}\sigma$ is normal with mean $\delta_0 \sqrt{n}$. This means that the power of our test will be the survival function of a $2n-2$ degree non-central T distribution with noncentrality parameter $\delta_0 \sqrt{n}$ evaluated at $t_{1-\alpha}$. We can find the number of samples $n$ that produces a desired power $\beta$ with a simple binary search. This calculation is implemented within the `statsmodels` library as follows:


```python
from statsmodels.stats import power
from scipy.optimize import brentq
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme()
```


```python
powers = [0.8, 0.85, 0.9]
```


```python
freq_df_t = pd.concat([pd.DataFrame({
    'samples': [2 * power.tt_ind_solve_power(effect_size=delta, alpha=alpha, power=p, alternative='larger') for p in powers],
    'power': powers, 'delta': delta, 'alpha': alpha})
    for delta in [0.01, 0.015, 0.02]
    for alpha in [0.2, 0.15, 0.1]], ignore_index=True)
```


```python
for ax in sns.relplot(data=freq_df_t, x='power', y='samples', hue='delta', col='alpha', kind='line').axes.ravel():
    ax.set_xticks(powers)
```



![png](/assets/images/sample_output_7_0.png)



This suggests we'd need a truly massive number of samples! Fortunately, things aren't as bad as they seem. We can use a Z-test instead.

Say response rates are $\theta_0$ and $\theta_1$ with difference $\delta$. The difference of observed arm means $\hat{\delta}$ will be approximately normal around $\delta$ with variance $\frac{\theta_0(1-\theta_0) + \theta_1(1-\theta_1)}{n}$. Let $\sigma_0^2 =2\theta_0(1-\theta_0)$ and  $\sigma_1^2 =\theta_0(1-\theta_0) + \theta_1(1-\theta_1)$.  if $\delta = 0$ and we reject when $\hat{\delta} > \frac{\sigma_0}{\sqrt{n}} z_{1 - \alpha}$, we'll have a false positive rate of $\alpha$. If $\delta = \delta_0$, on the other hand, then $\hat{\delta} - \delta_0$ will be normal with variance $\sigma_1^2/n$, so rejecting if $\hat{\delta} > \delta_0 + \frac{\sigma_1}{\sqrt{n}} z_{1 - \beta}$ would give us power $\beta$. We can choose $n$ so that these critical values are the same.

A little algebra tells us that $n = \frac{(\sigma_0 z_{1 - \alpha} - \sigma_1 z_{1 - \beta})^2}{\delta_0^2}$. Once again, this is already implemented in `statsmodels`.


```python
t = 0.04
```


```python
freq_df_z = pd.concat([pd.DataFrame({
    'samples': [2 * power.normal_sample_size_one_tail(d, p, alpha,
                        std_null=np.sqrt(2 * t * (1-t)),
                        std_alternative=np.sqrt(t * (1-t) + (t+d)*(1 - (t+d))))
                for p in powers],
    'power': powers, 'delta': d, 'alpha': alpha})
    for d in [0.01, 0.015, 0.02]
    for alpha in [0.2, 0.15, 0.1]], ignore_index=True)
```


```python
for ax in sns.relplot(data=freq_df_z, x='power', y='samples', hue='delta', col='alpha', kind='line').axes.ravel():
    ax.set_xticks(powers)
```



![png](/assets/images/sample_output_12_0.png)



## Frequentist Multi-Arm Estimation

With more than two arms, we end up testing more than one null hypothesis. Specifically, we'll want to test the null hypothesis that arm $i$ has a higher mean than arm $j$ for every pair arms $i,j$. We'll reject this hypothesis when $\hat{\delta_{i,j}} > c$ for a specific value of $c$. Our family-wise false positive rate $1 - \alpha$ must now cover the probability that we mistakenly reject any of these tests. If $\hat{\delta}_{i,j} > c$ for any $i,j$, it will also be true that $\delta^\star > c$ where $\delta^\star$ is the largest observed difference in means between any pair of arms. We can show that $\delta^\star$ follows a known distribution.

In general, if $F$ is a distribution's cdf, the joint probability that each of $k$ independent draws from the distribution are between $a$ and $a+\delta$ is given by $(F(a + \delta) - F(a))^k$. This makes the density at $a$ given by $k f(a)(F(a + \delta) - F(a))^{k-1}$. We want to integrate over all $a$: $\int_{-\infty}^\infty k f(a)(F(a + \delta) - F(a))^{k-1}\, da$.

This distribution over the largest difference within $k$ samples is known as the *Studentized range distribution* when $F$ is the cdf of a Student's t distribution. This is built into `scipy`! And because a normal distribution is just a Student's t distribution with infinite degrees of freedom, this covers a largest difference between normal samples as well, which I'll refer to as a *range distribution*.

Say each of the per-arm differences are normal with approximately the same variance $\sigma^2$. We can reject the null hypothesis that all the inter-arm differences are zero when $\delta^\star > \frac{\sigma}{\sqrt{n}} q_{1-\alpha}$ to get a family-wise false positive rate of $\alpha$, where $q_{1-\alpha}$ gives the $1-\alpha$ quantile of the range distribution. If we also reject when $\delta^\star > \delta_0 +  \frac{\sigma}{\sqrt{n}} q_{1-\beta}$, we'd have power $\beta$ when the true maximum difference is $\delta_0$. Once again, we can solve for when these critical values are equal to find the necessary number of samples; unfortunately the algebra here isn't built into `statsmodels`, so we'll have to do it ourselves.


```python
def range_sample_size(k, theta=0.06, delta=0.015, alpha=0.90, beta=0.8):
    q0 = stats.studentized_range.ppf(1 - alpha, k, np.inf)
    q1 = stats.studentized_range.ppf(1 - beta, k, np.inf)
    theta1 = theta + delta
    return k * theta1 * (1 - theta1) * ((q1 - q0) / delta)**2
```


```python
ks = range(2, 5)
```


```python
freq_df_q = pd.concat([pd.DataFrame({
    'samples': [range_sample_size(k, delta=delta, theta=theta, alpha=alpha) for k in ks],
    'k': ks, 'delta': delta, 'theta': theta, 'alpha': alpha})
    for delta in [0.01, 0.015, 0.02]
    for theta in [0.04, 0.05]
    for alpha in [0.2, 0.15, 0.1]], ignore_index=True)
```


```python
for ax in sns.relplot(data=freq_df_q, x='k', y='samples', hue='delta', col='theta', row='alpha', kind='line').axes.ravel():
    ax.set_xticks(list(ks))
```



![png](/assets/images/sample_output_19_0.png)



## Bayesian Estimation

Extending the reasoning in *A Bayesian Estimator of Sample Size* (Bi & Ji 2024), we can also pursue a Bayesian approach.
Say $p_1, p_2, \dotsc p_k \sim \text{Beta}(a,b)$. Assume that we observe mean $\theta - \delta$ from every arm except arm $k$, which has mean $\theta$. Say every arm got $n$ observations. It's easy to get a Monte Carlo estimate of the posterior probability that $p_k$ was the maximum sample.


```python
def max_arm_prob(n, theta=0.04, delta=0.02, k=3, s=5_000):
    theta2 = theta - delta
    sample = np.random.beta(4 + theta * n, 96 + (1 - theta) * n, size=s)
    return (stats.beta.cdf(sample, 4 + theta2 * n, 96 + (1 - theta2) * n)**(k-1)).mean()
```

Using this estimate, we can find the value of $n$ that would allow us to assume $p_k$ was maximal with 90% confidence. Although this isn't really a power, I'll refer to the confidence level as $\beta$ for ease of comparison to the previous plots.


```python
def binary_search(beta=0.9, f=max_arm_prob, **kwargs):
    try:
        return brentq(lambda n: f(n, **kwargs) - beta, 2, 2000, xtol=1)
    except:
        return np.nan
```


```python
bayes_df = pd.concat([pd.DataFrame({
    'samples': [k * binary_search(k=k, delta=delta, theta=theta, beta=beta) for k in ks],
    'k': ks, 'delta': delta, 'theta': theta, 'beta': beta})
    for delta in [0.01, 0.015, 0.02]
    for theta in [0.04, 0.05]
    for beta in [0.8, 0.9]], ignore_index=True)
```


```python
for ax in sns.relplot(data=bayes_df, x='k', y='samples', hue='delta', col='theta', row='beta', kind='line').axes.ravel():
    ax.set_xticks(list(ks))
```



![png](/assets/images/sample_output_26_0.png)



# Choosing Survey Questions

Consider now a related question where each of $n$ survey participants from a much larger target population will choose a response among $k$ alternatives. If we want to be able to safely identify which response is the most popular in our target population, how large can we make $k$?

## A Bayesian Approach

We want to find the posterior probability that the option that recieved the most votes in our sample is really the most popular in our full population. Say the most popular option $i$ in our survey recieved fraction $\frac{1 + \delta}{k}$ of the votes, and the rest were split equally among the other alternatives. If we started with a flat Dirichlet prior over for option popularity, it's easy to compute the posterior and sample from it.


```python
def max_option_prob(n, delta=0.01, k=3, s=5_000):
    theta = (1 + delta) / k
    theta2 = (1 - theta) / k
    counts = np.concatenate([np.full(1, theta), np.full(k-1, theta2)]) * n + 1
    ss = np.random.dirichlet(counts, size=s)
    return (ss[:,0] > ss[:, 1:].max(axis=1)).mean()
```


```python
ks = range(2, 10)
```


```python
survey_bayes_df = pd.concat([pd.DataFrame({
    'samples': [binary_search(k=k, delta=delta, f=max_option_prob, beta=beta) for k in ks],
    'k': ks, 'delta': delta, 'beta': beta})
    for delta in [0.01, 0.02]
    for beta in [0.8, 0.9]], ignore_index=True)
```


```python
for ax in sns.relplot(data=survey_bayes_df, x='k', y='samples', hue='delta', col='beta', kind='line').axes.ravel():
    ax.set_xticks(list(ks))
```



![png](/assets/images/sample_output_34_0.png)




```python

```
