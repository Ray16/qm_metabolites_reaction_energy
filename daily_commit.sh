#!/usr/bin/env bash
# Commit whatever is in the working tree, once a day, so uncommitted work is
# never more than a day old.
#
# Written after 2026-08-04, when pipeline/ was deleted from the worktree and
# ten untracked files from 07-30 were lost permanently because they had never
# been committed. Tracked files were recoverable; untracked ones were not.
#
# The guard matters as much as the commit. An unguarded "git add -A" run
# against a damaged tree would have committed that deletion into history,
# turning a recoverable accident into a permanent one. So this refuses to run
# when the tree looks damaged and leaves a loud note instead.
#
# Pushes by default (DAILY_COMMIT_PUSH defaults to 1) so the remote stays in
# sync -- a local-only commit does not survive disk loss and the remote was
# found 3 commits behind on 2026-08-14. Set DAILY_COMMIT_PUSH=0 to disable.
# Push is best-effort: if credentials are unavailable it logs a WARN and the
# commit is still safe locally. (Credentials verified working for HTTPS push.)

set -uo pipefail

REPO="${DAILY_COMMIT_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LOG="${DAILY_COMMIT_LOG:-$REPO/.daily_commit.log}"
MAX_DELETIONS="${DAILY_COMMIT_MAX_DELETIONS:-10}"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$LOG"; }

cd "$REPO" || { log "FATAL cannot cd to $REPO"; exit 1; }

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    log "FATAL $REPO is not a git repository"
    exit 1
fi

# Never commit on top of an interrupted operation; the tree is not the user's.
git_dir=$(git rev-parse --git-dir)
for state in rebase-merge rebase-apply MERGE_HEAD CHERRY_PICK_HEAD BISECT_LOG; do
    if [ -e "$git_dir/$state" ]; then
        log "SKIP $state in progress; leaving the tree alone"
        exit 0
    fi
done

if [ -z "$(git status --porcelain)" ]; then
    log "ok  nothing to commit"
    exit 0
fi

# Damage guard: a burst of deleted tracked files means something removed them,
# not that the user intended it. Report instead of recording it.
deleted=$(git ls-files --deleted | wc -l | tr -d ' ')
if [ "$deleted" -gt "$MAX_DELETIONS" ]; then
    log "REFUSED $deleted tracked files are missing from the worktree (limit $MAX_DELETIONS)."
    log "        Nothing was committed. Recover with: git restore <paths>"
    log "        Override once with DAILY_COMMIT_MAX_DELETIONS=<n> if the deletions are real."
    git ls-files --deleted | sed 's/^/            /' >>"$LOG"
    exit 2
fi

git add -A
if git diff --cached --quiet; then
    log "ok  only ignored changes; nothing staged"
    exit 0
fi

files=$(git diff --cached --name-only | wc -l | tr -d ' ')
summary=$(git diff --cached --shortstat | sed 's/^ *//')

if git commit -q -m "Daily snapshot $(date '+%Y-%m-%d')" \
    -m "Automatic commit of in-progress work by daily_commit.sh: ${summary:-no diffstat}."; then
    log "ok  committed $files file(s): ${summary:-n/a}  -> $(git rev-parse --short HEAD)"
else
    log "FAIL git commit returned non-zero"
    exit 1
fi

if [ "${DAILY_COMMIT_PUSH:-1}" = "1" ]; then
    if git push -q 2>>"$LOG"; then
        log "ok  pushed to $(git remote get-url origin 2>/dev/null)"
    else
        log "WARN push failed (credentials?); the commit is safe locally"
    fi
fi
