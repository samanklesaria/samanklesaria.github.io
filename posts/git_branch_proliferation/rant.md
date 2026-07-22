---
title: "Delete your local branches"
date: "2026-06-11"
categories: "tools"
---

I create a branch every time I open a PR, and I end up with dozens of stale local branches. It's a slow accumulation — each one feels harmless, but eventually `git branch` scrolls off the screen and I can't find anything.

## The usual fix

I can periodically purge merged branches with [`git delete-merged-branches`](https://github.com/tj/git-extras/blob/main/Commands.md#git-delete-merged-branches). It works, but it's a chore I have to remember to do, and by the time I think of it, the clutter is already bad.

## Deleting immediately after pushing

Once I've pushed a branch to the remote for a PR, why keep it locally? It's saved up there. I just delete it and switch back to `main`:

```bash
git push origin my-feature
git switch main
git branch -D my-feature
```

This keeps my local repo clean at all times, with zero periodic maintenance. I've saved this in a script called `git-shove`, so this is all as easy as a push.

## What about review iterations?

The obvious problem: someone reviews my PR and requests changes. I need the branch back to push fixes.

But review can take days. In the meantime, I'm working on other things with a cleaner branch list. When I do need to revisit the PR, I pull it down with [`git pr`](https://github.com/tj/git-extras/blob/main/Commands.md#git-pr):

```bash
git pr 42
```

This checks out the PR branch directly from the remote. I make my changes, push, and delete the local branch again.

## The feeling

There's something deeply satisfying about running `git branch` and seeing just `main`. It's the git equivalent of a clean desk. I know it doesn't actually matter. I know I could just ignore the clutter. But every time I see that single line of output, a small, irrational part of my brain thinks: yes. Order. I am in control of at least this one thing.
