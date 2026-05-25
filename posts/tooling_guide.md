---
title: "An Opinionated Tooling Guide"
date: "2025-06-01"
categories: ["tools"]
image: https://upload.wikimedia.org/wikipedia/commons/4/44/Drawing_Tools_Flat_Icon_Vector.svg
---

**Statistics and Data Analysis:** Overall: use R. It has the largest ecosystem of statistical packages. 

- For Bayesian modeling, Stan has bindings for everything, but it's easiest to use from R, and the rest of the toolkit *(brms, loo, bayesplot, etc)* doesn't have an equivalents outside of R. Turing.jl and PyMC3 are substantially slower, and Numpyro is less mature (although it can be faster on a GPU). 
- For DiD studies, R has `did` and Python has `diffindiff` and `moderndid`, but Julia doesn't have anything comparable. 
- For visualization, R has ggplot2 and ggpairs, Julia has AlgebraOfGraphics and Python has Seaborn. They're all about as good as each other. 
- The data manipulation, R has the tidyverse, Python has polars and Julia has Dataframes.jl. I actually find DataFrames.jl to be the nicest of the bunch (with DataFramesMeta). Polars makes you learn a whole separate DSL instead of using the base language, and the rest of the Python ecosystem uses Pandas, so you constantly have to convert back and forth. Pandas itself its a nonstarter - generally slow, verbose, and hard to use. 
- For geographic queries, R's *sf* and Julia's *GeoStats* are lovely. Python's *GeoPandas*, being based on Pandas, is much more annoying to use. 
- For frequentist statistics, Python's *statsmodels* and Julia's *GLMs, CovarianceMatrices.jl, MixedModels* and *Survival* packages give you much of base R and the *lmer, sandwich* and *survival* packages.
- For Gaussian Process modeling, Python's *GPytorch* is far faster and more feature-complete than anything else. Julia's *AbstractGPs* has a nicer interface and is easier to extend, but it's slower and lacks the approximate inference algorithms necessary for large datasets.

**Neural Nets**

- There's a plethora of SAAS companies that promise easier experiment tracking. But in my experience, the easiest option is just to use git worktrees. Each experiment should live in a separate branch with a README documenting exactly what's being tested. This makes it easy to fork and cherry-pick pieces of different experiments. 
- For training, use Jax with equinox. 
- For visualization, use Tensorboard. 

**Other Programming**

- Use Julia for anything numerical.
- Use Python with the `fire` package, the `pytest-watch` test runner and the `uv` package manager for scripting.
- Use awk for text processing and jq for JSON processing
- Use Rust if efficiency matters.
- Otherwise, use Haskell with `ghcid`. 

**Notebooks:** I currently use Pluto for Julia, Marimo for Python, and RStudio (with qmd files) for R. LLMs work best on Markdown variants rather than structured notebook formats 

**Static Site Generation:** Use Quarto. Compared to Jekyll, niceties like latex math, computational notebooks and bibliography management are built in.

**Editors::** 
- Kakoune is lightweight and but has full lsp support. On linux, it's my perferred editor. But it assumes that window management functionality will be handled by your window manager, which isn't really a thing on other platforms.
- On a Mac, I use Zed. 

**Other**

- Use Xmonad on linux.
- Zulip is much nicer than Slack.
- I write down everything I might want to remember later in Anki. It's effectively a notes app for me.  
- Zotero has remained fast for me despite adding gigabytes of papers to its library. 
- Pixi is essential for package management across languages.
