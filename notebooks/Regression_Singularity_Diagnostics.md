---
title: Diagnosing Lack of Independence in Exogenous Variables
date: 05/06/2024
category: statistics
summary: This post outlines a simple workflow for diagnosing lack of independence in `statsmodels`.
---

While performing linear regression with `statsmodels`, you might occasionally find that your exogenous variables aren't independent, giving you a error about a singular matrix.

To figure out exactly which variables are colinear, I tend to use the following recipe:

1. Take the SVD of the design matrix $X = QSV^T$.
2. Find a column of $V$ that corresponds to a zero singular value.
3. Check which terms in our original formula correspond to the nonzero elements of $V$. Usually there's only a couple nonzero terms.

For posterity, I've reproduced the workflow below.

```python
m = dmatrix(formula, df)
u, s, vh = np.linalg.svd(m)
misfits = (np.abs(vh[s < 1e-8]) > 1e-5)
np.array(m.design_info.column_names)[misfits[0]]
```
