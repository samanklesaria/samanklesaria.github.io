---
title: "An Opinionated Tooling Guide"
date: "2025-06-01"
categories: ["tools"]
---

**Statistics and Data Analysis:** Overall: use R. Its has the largest ecosystem of statistical packages. 

- For Bayesian modeling, Stan has bindings for everything, but it's easiest to use from R, and the rest of the toolkit *(brms, loo, bayesplot, etc)* doesn't have an equivalent outside of R. Turing.jl and PyMC3 are substantially slower, and Numpyro is less mature (although it can be faster on a GPU). 
- For visualization, R has ggplot2 and ggpairs, Julia has AlgebraOfGraphics and Python has Seaborn. They're all about as good as each other. 
- The data manipulation, R has the tidyverse, Python has polars and Julia has Dataframes.jl. I actually find Dataframes.jl to be the nicest of the bunch (with DataframesMeta). Polars makes you learn a whole separate DSL instead of using the base language, and the rest of the Python ecosystem uses Pandas, so you constantly have to convert back and forth. Pandas itself its a nonstarter - generally slow, verbose, and hard to use. 
- For geographic queries, R's *sf* and Julia's *GeoStats* are lovely. Python's *GeoPandas*, being based on Pandas, is much more annoying to use. 
- For frequentist statistics, Python's *statsmodels* and Julia's *GLMs, MixedModels*, *Survey* and *Survival* packages give you much of base R and its *lmer*, *survey* and *survival* packages. 
- For Gaussian Process modeling, Python's *GPytorch* is far faster and more feature-complete than anything else. Julia's *AbstractGPs* has a nicer interface and is easier to extend, but it's slower and lacks the approximate inference algorithms necessary for large datasets.

**Neural Nets**

- There's a plethora of SAAS companies that promise easier experiment tracking. But in my experience, the easiest option is just to use git worktrees. Each experiment should live in a separate branch with a README documenting exactly what's being tested. This makes it easy to fork and cherry-pick pieces of different experiments. Use git-lfs for weights and training data. 
- For training, use Jax with equinox. 
- For visualization, use Tensorboard. 

**Other Programming**

- Use Julia for anything numerical
- Use Python for scripting
- Use awk for text processing and jq for JSON processing
- Use Rust if you want to be super-efficient. 
- Otherwise, use Haskell. 

**Notebooks:** I use Pluto for Julia, Marimo for Python, and Jupyter for everything else. 

**Static Site Generation:** Use Quarto. Compared to Jekyll, niceties like latex math, computational notebooks and bibliography management are built in.

**Databases**

- For OLTP, use postgres with pgvector & pgvectorscale for vector search and postGIS for spatial data. 
- For analytical workloads, use DuckDB
- For local RAGs, use ChromaDB. 

**Other**

- Zulip is much nicer than Slack.
- I write down everything I might want to remember later in Anki. It's effectively a notes app for me.  
- Zotero has remained fast for me despite adding gigabytes of papers to its library. 
- Ollama is great for local LLMs.
- Pixi is essential for package management across languages.
