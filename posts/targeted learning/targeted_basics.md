---
title: "Targeted Learning Basics"
date: "2026-06-05"
categories: "statistics"
image: https://tlverse.org/tlverse-handbook/img/logos/vdl-logo-transparent.svg
---

What follows is a highly informal and simplified introduction to targeted learning. Our goal will be to estimate some functional $T$ of a distribution $P$ from a collection of its samples. I'll denote the empirical (discrete) distribution given by the first $n$ samples as $P_n$, and I'll write expectations $\int f\, dP$ as $Pf$. When the distribution in question is clear, I'll use $E[f]$ for $Pf$ and $E_n[f]$ for $P_nf$. 

Our first step will be to use some other estimator to guess $T(P)$. We can train a machine learning algorithm on the samples in $P_n$ to get a guess $\hat{P}$ of the true distribution, and then make the guess $T(\hat{P})$. 

Once we have an initial estimate, the targeted learning framework comes in. Our estimator $T(\hat{P})$ is said to be *asymptotically linear* if
$$
\sqrt{n}(T(\hat{P} - T(P)))= n^{-1/2}P_nL_{T(P)} + o_p(1)
$$
where the $o_p(1)$ converges in probability to zero as $n \to \infty$. The function $L_{T(P)}$ is the  *influence function* for $T(P)$: a kind of distributional-derivative. Specifically:
$$
L_{T(P)}(x) = \frac{\partial T(\epsilon \delta_x + (1 - \epsilon)P)}{\partial \epsilon}
$$
where $\delta_x$ is the delta distribution at $x$. If $T(\hat{P})$ is asymptotically linear, the definition above suggests that either:

1. We can get a better estimator for $T(P)$ by using $T(\hat{P}) - P_n L_{T(P)}$ instead. This is known as the *one step* estimator. 

2. We should should refine our guess $\hat{P}$ to some $\hat{P}^*$ so that $P_n L_{T(\hat{P}^*)}=0$. This is known as *targeted maximum likelihood estimation* or TMLE. 

   

**Example 1: Mean Estimates**

If you're estimating $T(P) = E[Y]$, the influence function $L(y)$ is $y - PY$. We approximate $PY \approx P_nY$. But this means that $P_n L = 0$ already. The targeting step doesn't change anything. We just get the plug in estimator. 



**Example 2: Conditional Means**

If you're estimating $T(P) = E[Y | A=1]$ without confounders, the influence function $L(a,y) = \frac{1_{a=1}}{P(A=1)}(y - E[Y | A=1])$. If we approximate $P(A=1) \approx P_n(A=1)$ and $E[Y | A=1] \approx E_n[Y | A=1]$, we once again see $E_n[L] = 0$. 



**So when do things get interesting?**

Targeted estimation only starts having an effect when our estimators for nuisance parameters start telling us something different from what we get by looking at sample means. Consider when we're estimating $E[Y | A=1]$ with confounders $W$. Then
$$
\begin{align*}
L(y,a,w)&= \frac{a}{P(A=1 | W=w)}(Y - E[Y | W=w, A=1]) + \\
&E[Y | W=w, A=1] - E[Y | A=1]
\end{align*}
$$
If our approximations $Q(w)$ of $E[Y | W=w, A=1]$ and $g(w) = P(A=1|W=w)$ are complicated machine learning estimators rather than empirical means, the cancellation we've seen so far doesn't occur. 