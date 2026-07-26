#!/usr/bin/env bash
# Re-sync the Overleaf-facing branch from the paper/ directory of this branch.
#
#   scripts/sync_overleaf.sh              # update the branch locally
#   scripts/sync_overleaf.sh --push       # ...and push it to origin
#
# `overleaf-paper-unbias` holds *only* the contents of paper/, with main.tex at
# the repository root, which is the layout Overleaf's git integration expects.
# It is produced by `git subtree split`, so it is a genuine projection of this
# branch's history rather than a copy: every commit that touched paper/ appears,
# nothing that didn't does, and re-running is idempotent.
#
# Run this after committing paper changes on the source branch. Regenerate the
# figures first if results have moved:  scripts/make_figures.py
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH=${BRANCH:-overleaf-paper-unbias}
PREFIX=${PREFIX:-paper}
SOURCE=$(git branch --show-current)

if [ "$SOURCE" = "$BRANCH" ]; then
  echo "refusing to run from $BRANCH itself; switch to the source branch first" >&2
  exit 1
fi

if ! git diff --quiet -- "$PREFIX" || ! git diff --cached --quiet -- "$PREFIX"; then
  echo "uncommitted changes under $PREFIX/ — commit them first, or the split" >&2
  echo "will silently publish the last committed version instead:" >&2
  git status --short -- "$PREFIX" >&2
  exit 1
fi

echo "splitting $PREFIX/ out of $SOURCE ..."
SPLIT=$(git subtree split --prefix="$PREFIX")
git branch -f "$BRANCH" "$SPLIT"
echo "$BRANCH -> $(git log --oneline -1 "$BRANCH")"
echo "root:"
git ls-tree --name-only "$BRANCH" | sed 's/^/  /'

if [ "${1:-}" = "--push" ]; then
  if git remote get-url origin >/dev/null 2>&1; then
    git push --force-with-lease origin "$BRANCH:$BRANCH"
  else
    echo "no 'origin' remote configured; add one before --push" >&2
    exit 1
  fi
fi
