---
name: commit
description: Split the current uncommitted changes into multiple separate git commits, each with a conventional-commit type prefix (feat/fix/perf/refactor/docs/test/chore/style/build/ci). Use when the user asks for their pending work to be committed as several commits instead of one, asks for commits "prefixed with feat/fix/chore etc.", or asks for a conventional-commit-style history.
---

# Conventional commits (multi-commit split)

Use this when the user wants pending changes committed as **more than one**
commit, each labeled with a conventional-commit type — not a single
`git add -A && git commit`.

## 1. Read the actual diff before deciding anything

Run `git status` and `git diff` (or review the files you just wrote this
session) before proposing a split. Never guess at groupings from memory of
what you intended to do — confirm what actually changed on disk.

## 2. Group by concern, not by chronology or file listing order

A good split answers "if someone reverted just this commit, would that
make sense as one undo?" Typical groupings, in the order they usually
belong (earlier commits shouldn't depend on later ones):

- **`chore`/`docs`**: project scaffolding, config manifests (`pyproject.toml`,
  `package.json`, `.gitignore`), README, design/architecture docs, decision
  logs, prompt/audit records. Prose documentation files are hard to split
  by line without fragility — if a single doc file's content narrates
  several unrelated decisions, don't try to slice it; commit its current
  full state here rather than forcing it to mirror the code commits.
- **`feat`**: new functionality and the tests that cover it (tests travel
  with the feature they test, not into a separate `test` commit, unless
  the user asks for tests split out specifically).
- **`fix`**: correction of a real bug in already-existing (even if
  uncommitted-this-session) behavior.
- **`perf`**: a deliberate performance/efficiency change to working code —
  quantify it in the commit body if you have real numbers (before/after
  timings, sizes), not just an assertion that it's faster.
- **`refactor`**: restructuring without a behavior change.
- **`style`/`build`/`ci`**: as applicable, rarely relevant to this kind of split.

Pick the type that's honest about the change's *nature*, not whichever
type sounds most impressive. A file introduced for the first time can
still be a `perf` or `fix` commit if the reason it exists is a performance
or correctness concern — "was there a prior commit to diff against" is not
the test; "what is this commit's defining characteristic" is.

## 3. When one file mixes concerns that belong in different commits

This is the common hard case: e.g. a feature file that later got a small
optimization added to it, and you want the optimization in its own `perf`
commit separate from the `feat` commit, but nothing has been committed yet
so there's no prior git history to diff against.

**Do not** force an artificial split (e.g. dumping unrelated files into a
commit just to make three commits exist) and **do not** fabricate commit
messages that misrepresent what's actually in each commit.

**Do** reconstruct the two states honestly:

1. Identify exactly which lines belong to the later concern (you likely
   just wrote them this session — you know the diff precisely).
2. Temporarily revert just those lines/hunks in the working tree (via Edit,
   not `git checkout`, since there's nothing committed yet to restore from).
3. Run the test suite. Confirm the reverted state is genuinely a coherent,
   working baseline — not a state you're inventing that never actually ran.
4. Stage and commit that baseline under its true type (usually `feat`).
5. Re-apply the reverted lines exactly (you have them from step 1).
6. Run the test suite again.
7. Stage and commit just those files under their true type (usually `fix`
   or `perf`), with a commit body that explains the concrete before/after
   (numbers if you measured them).

This produces an honest, meaningful history instead of a cosmetic 3-way
split of one big blob.

## 4. Stage explicitly, never broadly

Use `git add <specific files>` per commit. Never `git add -A` / `git add .`
when splitting — that defeats the entire point and risks silently pulling
unrelated changes into the wrong commit. Run `git status` before each
commit to confirm exactly what's staged, and `git diff --cached` if there's
any doubt about a file that might contain mixed concerns.

## 5. Verify each commit independently

Run the project's test suite after staging each commit's content, before
committing it — not just once at the very end. Each commit in the sequence
should be a state that actually works, in case someone bisects later.

## 6. Commit message format

Subject line: `<type>: <imperative summary>`, lowercase after the colon,
no trailing period. A body is welcome and encouraged when there's real
substance to explain (why, not just what — the file diff already shows
what). Follow the repo's existing commit style if there's prior history to
match; otherwise this format is the sane default.

## 7. Finish by showing the result

Run `git log --oneline -n <count>` and `git status` after the last commit
so the user can see the resulting history and confirm the working tree is
clean.
