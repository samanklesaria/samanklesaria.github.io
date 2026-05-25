---
title: "Case Control Studies"
date: 5/20/2026
image: img.png
categories: [statistics]
---

Say we want to find the odds ratio for how a factor $X$ affects the probability of having a disease $D$. We can model this with logistic regression:
$$
\text{logit}(P(D=1| X)) = \alpha + \beta X
$$
But if the disease is sufficiently rare, sampling everyone and waiting for cases is impractical. We need to oversample the population that has the disease. Let $S=1$ for units we sample. Let $\pi_1 = P(S=1 | D = 1)$ and $\pi_0 = P(S=1 | D = 0)$. How do we change our regression for account for the unequal sampling probabilities?



### Perspective 1

If we use logistic regression on our oversampled dataset, the coefficient $\beta$ we get actually estimates $\log \frac{\text{odds } D | X = 1, S=1}{\text{odds }D | X=0, S=1}$. But once you condition on $X$, the odds of $D$ don't change with $S$. By Bayes' Rule:
$$
\frac{P(D =1 | X, S=1)}{P(D=0 | X, S=1)} = \frac{\pi_1P(D =1 | X)}{\pi_0P(D=0 | X)}
$$
The $\pi_1/\pi_0$ factor cancels out when we divide the odds for $X=1$ by the odds of $X=0$. So the $\beta$ coefficient we get from the oversampled dataset represents the true log odds ratio! No adjustment to the estimate or its standard errors are necessary!



### Perspective 2

The standard way to dealt with unequal sampling probabilities is to weigh observations where $D=i$ by $1 / \pi_i$ when doing maximum likelihood estimation. This is equivalent to us fitting the model
$$
\text{logit}(P(D| X)) = \log \pi_1/\pi_0 + \alpha + \beta X
$$
The $\log \pi_1/\pi_0$ part will get rolled into our estimate of the intercept, but our the coefficient for $\beta$ will stay the same. 



### How many should we sample?

So now we know that it's fine to sample the $D=1$ and $D=0$ populations separately. How many samples should we take from each population?

For simplicity, we can work with binary $X \in \{0,1\}$ for a clean closed form. The case-control data form a 2×2 table with cells $n_1 p_1,\, n_1(1-p_1),\, (n - n_1) p_0,\, (n - n_1)(1-p_0)$, where $p_j = P(X=1 \mid D=j)$. By the delta method on $\hat\beta = \log\hat{OR}$:
$$
\operatorname{Var}(\hat\beta) = \frac{1}{n_1 p_1} + \frac{1}{n_1(1-p_1)} + \frac{1}{n_0 p_0} + \frac{1}{n_0(1-p_0)} = \frac{1}{n_1 V_1} + \frac{1}{n_0 V_0}
$$
where $V_j = p_j(1-p_j) = \operatorname{Var}(X \mid D=j)$. Deriving the variance wrt $n_1$ for a fixed total sample size gives $\frac{n_1}{n_0} = \sqrt{\frac{V_0}{V_1}}$. So: we should sample group $i$ in proportion to $\sqrt{V_i}$.
