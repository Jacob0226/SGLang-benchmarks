#!/usr/bin/env bash
# Build a RAID0 array from every IDLE NVMe drive on the box (auto-detected)
# and mount it at /raid for use as the HiCache L3 directory.
#
# SAFETY:
#   * The boot/root drive is auto-detected (via the disk backing / and
#     /boot/efi) and NEVER included -- we do NOT hard-code a device number,
#     because Linux nvmeN numbering follows PCIe enumeration order, not boot
#     order (e.g. on this box the boot drive is nvme7, not nvme0).
#   * Only drives with NO partition table, NO filesystem, and NO mountpoint
#     are eligible -- any drive that already holds data is refused.
#   * RAID0 has NO redundancy: a single drive failure loses the whole array.
#     That is fine for a scratch KV-cache (L3) tier — it is recomputable.
#
# Run with sudo:  sudo bash tools/setup_raid0_l3.sh
#   override the auto-detected member list with:  DRIVES="/dev/nvmeXn1 ..." sudo -E bash tools/setup_raid0_l3.sh
set -euo pipefail

MD=/dev/md0
MNT=/raid
OWNER=${SUDO_USER:-jacchang}

# ---- detect the boot/root disk(s) so we never touch them ----
# Map the filesystem source of / and /boot/efi back to their parent whole
# disk (PKNAME), e.g. /dev/nvme7n1p2 -> nvme7n1. Anything in this set is
# excluded from the RAID0 regardless of its device number.
declare -A BOOT_DISKS=()
for mp in / /boot /boot/efi; do
  src=$(findmnt -no SOURCE "$mp" 2>/dev/null) || continue
  [[ -n "$src" ]] || continue
  pk=$(lsblk -no PKNAME "$src" 2>/dev/null | head -1)
  [[ -z "$pk" ]] && pk=$(basename "$src")   # source is already a whole disk
  [[ -n "$pk" ]] && BOOT_DISKS["$pk"]=1
done
echo ">>> boot/root disk(s) auto-detected & EXCLUDED: ${!BOOT_DISKS[*]:-<none?>}"

# ---- build the candidate member list ----
# Honor a caller-supplied DRIVES override; otherwise auto-collect every whole
# NVMe disk that is not a boot disk, has no filesystem, and is not mounted.
if [[ -n "${DRIVES:-}" ]]; then
  read -r -a DRIVES <<< "$DRIVES"
  echo ">>> using caller-supplied DRIVES override"
else
  DRIVES=()
  while read -r disk; do
    [[ -n "${BOOT_DISKS[$disk]:-}" ]] && continue          # skip boot disk
    if lsblk -no FSTYPE "/dev/$disk" | grep -q .; then continue; fi   # skip if any fs
    if lsblk -no MOUNTPOINT "/dev/$disk" | grep -q .; then continue; fi  # skip if mounted
    DRIVES+=("/dev/$disk")
  done < <(lsblk -dn -o NAME,TYPE | awk '$2=="disk" && $1 ~ /^nvme/ {print $1}' | sort -V)
fi

[[ ${#DRIVES[@]} -ge 1 ]] || { echo "FATAL: no idle NVMe drives found to build the array"; exit 1; }
echo ">>> target drives (${#DRIVES[@]}): ${DRIVES[*]}"

# ---- safety checks (defense-in-depth; re-verify each member) ----
for d in "${DRIVES[@]}"; do
  name=$(basename "$d")
  [[ -n "${BOOT_DISKS[$name]:-}" ]] && { echo "FATAL: $d is a boot disk — aborting"; exit 1; }
  [[ -b "$d" ]] || { echo "FATAL: $d is not a block device"; exit 1; }
  if lsblk -no MOUNTPOINT "$d" | grep -q .; then
    echo "FATAL: $d (or a child) is mounted — aborting"; exit 1; fi
  if lsblk -no FSTYPE "$d" | grep -q .; then
    echo "FATAL: $d has a filesystem/partition — aborting (won't wipe data)"; exit 1; fi
done
echo ">>> all ${#DRIVES[@]} drives confirmed empty & unmounted."

# ---- create array ----
echo ">>> creating RAID0 $MD (chunk 512K)"
mdadm --create "$MD" --level=0 --chunk=512 --raid-devices=${#DRIVES[@]} "${DRIVES[@]}"

echo ">>> mkfs.ext4 (large-file tuned)"
mkfs.ext4 -F -m 0 -O ^has_journal "$MD"   # no journal: max write BW for scratch tier

mkdir -p "$MNT"
mount -o noatime "$MD" "$MNT"
chown "$OWNER":"$OWNER" "$MNT"

echo ">>> done. RAID0 mounted at $MNT"
df -h "$MNT" | tail -1
echo
echo "Optional persistence across reboot (review before running):"
echo "  mdadm --detail --scan | sudo tee -a /etc/mdadm.conf"
echo "  echo '$MD  $MNT  ext4  noatime,nofail  0 0' | sudo tee -a /etc/fstab"
echo
echo "Point HiCache L3 here when running cascade_dsr1_lite.sh:"
echo "  1) start the container with the array mounted in:   -v $MNT:$MNT"
echo "  2) run the bench with:   L3_BASE_DIR=$MNT ./cascade_dsr1_lite.sh ... --cache-modes L3_file"
