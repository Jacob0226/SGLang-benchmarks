#!/usr/bin/env bash
# Build the GLM-5.3-Flash MI355X PR stack on an image's SGLang commit.
#
# Each PR is applied as its `gh pr diff` (the PR's own change against its merge
# base), so main commits a PR branch merged in do not come along. One commit per
# entry; stops at the first entry that does not apply.
#
# usage: build_glm53_prstack.sh <image sglang commit> <worktree dir> [entry...]
#   entry: a sglang PR number, or c:<commit> for a local-only commit
#   default entries: the stack listed in AgentSkill/glm53-flash/pr-stack.md
# After resolving a conflict by hand: git add, commit, then rerun with the remaining entries.
set -euo pipefail

BASE=$1 WT=$2
shift 2
REPO=${REPO:-$HOME/PR/sglang}
GH="env -u GH_TOKEN -u GITHUB_TOKEN /home/jacchang/bin/gh"
GIT=(git -c user.name="Jacob0226" -c user.email="jacchang@amd.com")
# #44416 carries #44415's commit
ENTRIES=("$@")
[ ${#ENTRIES[@]} -eq 0 ] && ENTRIES=(41615 41871 38764 44416 44417 46808 44659 44661 48875)
LOCAL_TREE=${LOCAL_TREE:-/shared_nfs/jacchang/exp/sglang-fp8kv-1103}

cd "$REPO"
prs=()
for e in "${ENTRIES[@]}"; do [[ $e == c:* ]] || prs+=("pull/$e/head:refs/remotes/pr/$e"); done
git fetch -q -f upstream "${prs[@]}"
[ -d "$WT" ] || git worktree add -q -b "glm53-prstack-${BASE:0:10}" "$WT" "$BASE"
cd "$WT"

for e in "${ENTRIES[@]}"; do
    patch=$(mktemp)
    if [[ $e == c:* ]]; then
        c=${e#c:}
        git -C "$LOCAL_TREE" show --format= "$c" > "$patch"
        msg=$(git -C "$LOCAL_TREE" log -1 --format=%s "$c")
        # git apply -3 needs the commit's blobs in this repo
        git fetch -q "$LOCAL_TREE" "+refs/heads/*:refs/remotes/localtree/*"
    else
        $GH pr diff "$e" --repo sgl-project/sglang > "$patch"
        head=$(git rev-parse --short=10 "refs/remotes/pr/$e")
        msg="PR $e (head $head): $($GH pr view "$e" --repo sgl-project/sglang --json title --jq .title)"
    fi
    if git apply -3 --index "$patch" 2>/tmp/prstack_apply.err; then
        "${GIT[@]}" commit -q -m "$msg"
        echo "ok    $e"
    else
        echo "FAIL  $e"
        cat /tmp/prstack_apply.err
        git status --short | grep -E '^(UU|AA|DU|UD)' || true
        rm -f "$patch"
        exit 1
    fi
    rm -f "$patch"
done
git log --oneline "$BASE"..HEAD
