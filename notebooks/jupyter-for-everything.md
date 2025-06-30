---
title: Jupyter for Everything
category: tools
date: 08/20/2020
---

Forget vim or emacs of VSCode. [Jupyter](https://jupyter.org/) is hands down the best editor out there. The literate programming support, complete with images and beautiful latex snippets? The inline visualizations of all kinds of data-types, from [maps](https://ipyleaflet.readthedocs.io/en/latest/api_reference/map.html) to [meshes](https://pythreejs.readthedocs.io/en/stable/examples/Geometries.html) to [graphs](https://networkx.github.io/) and [polygons](https://shapely.readthedocs.io/en/stable/)?  The [widgets](https://ipywidgets.readthedocs.io/en/stable/examples/Using%20Interact.html) for interactively testing your code? It's glorious! I've listed examples from IPython, but IJulia and IHaskell have similar goodies, and there's even kernels for languages like C and Rust. And yet no one seems to use them outside of exploratory data analysis. 

Part of the problem, I think, is that other tools can't read Jupyter's file format. Normal editors like vim or VSCode edit text files, which can be tracked in version control systems or compiled and tested from build systems or checked by static analysis tools or executed by interpreters ... you get the idea. But there's no reason Jupyter has to use its default JSON file format. Why not make Jupyter read and save text files, like every other editor?

As it turns out, this is actually really easy to do. I made a package for it [here](https://github.com/samanklesaria/jupsource). Now, you can use Jupyter with any source file you want. Want to try out some interactive examples as you're reading through an unfamiliar codebase? Open it in Jupyter. Want to turn your exploratory programming session into a reusable module? It already is one! 

Using Jupyter for everything also means you don't have to separate the visualizations and exploratory snippets you used to derive your algorithm from its implementation in production. Generators of fake datasets, results on toy problems, experiments with different parameters- this is all useful stuff for people reading your code to know about, even if this code isn't being run. As Ableson famously said, "Programs must be written for people to read, and only incidentally for machines to execute".

To summarize: Jupyter notebooks shouldn't just be treated as a kind of fancy repl. They're useful at every stage of a project's life-cycle, from experimenting to prototyping to presenting to production. Delete your text editor. You won't be needing it anymore. 

