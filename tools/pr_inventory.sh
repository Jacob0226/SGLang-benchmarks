#!/bin/bash
# Inventory ~/PR: one TSV row per entry (name, kind, branch, last commit date, dirty files, unpushed commits, files).
cd ~/PR || exit 1
printf "name\tkind\tbranch\tlast_commit\tdirty\tunpushed\tfiles\n"
for d in */ *; do
  d=${d%/}
  [ -e "$d" ] || continue
  if [ -f "$d" ]; then
    printf "%s\tfile\t-\t%s\t-\t-\t1\n" "$d" "$(date -r "$d" +%F)"
    continue
  fi
  [ -d "$d" ] || continue
  case " $seen " in *" $d "*) continue ;; esac
  seen="$seen $d"
  files=$(find "$d" -xdev -type f -not -path "*/.git/*" 2>/dev/null | wc -l)
  if git -C "$d" rev-parse --git-dir >/dev/null 2>&1; then
    if [ -f "$d/.git" ]; then kind=worktree; else kind=repo; fi
    branch=$(git -C "$d" symbolic-ref --short -q HEAD || echo "detached@$(git -C "$d" rev-parse --short HEAD)")
    last=$(git -C "$d" log -1 --format=%cs 2>/dev/null)
    dirty=$(git -C "$d" status --porcelain --untracked-files=normal 2>/dev/null | wc -l)
    unpushed=$(git -C "$d" rev-list --count HEAD --not --remotes 2>/dev/null)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$d" "$kind" "$branch" "$last" "$dirty" "$unpushed" "$files"
  else
    printf "%s\tdir\t-\t%s\t-\t-\t%s\n" "$d" "$(date -r "$d" +%F)" "$files"
  fi
done
