#!/usr/bin/env bash
# Build a RAID0 array from the 7 idle Micron 7450 NVMe drives (nvme1..nvme7)
# and mount it at /raid for use as the HiCache L3 directory.
#
# SAFETY:
#   * nvme0 (boot/root) is NEVER included.
#   * Refuses any target drive that is mounted, has a partition table, or has a filesystem.
#   * RAID0 has NO redundancy: a single drive failure loses the whole array.
#     That is fine for a scratch KV-cache (L3) tier — it is recomputable.
#
# Run with sudo:  sudo bash tools/setup_raid0_l3.sh
set -euo pipefail

MD=/dev/md0
MNT=/raid
OWNER=${SUDO_USER:-jacchang}
DRIVES=(/dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 /dev/nvme5n1 /dev/nvme6n1 /dev/nvme7n1)

echo ">>> target drives: ${DRIVES[*]}"
echo ">>> EXCLUDED (boot): /dev/nvme0n1"

# ---- safety checks ----
for d in "${DRIVES[@]}"; do
  [[ "$d" == /dev/nvme0n1 ]] && { echo "FATAL: refusing nvme0 (boot)"; exit 1; }
  [[ -b "$d" ]] || { echo "FATAL: $d is not a block device"; exit 1; }
  if lsblk -no MOUNTPOINT "$d" | grep -q .; then
    echo "FATAL: $d (or a child) is mounted — aborting"; exit 1; fi
  if lsblk -no FSTYPE "$d" | grep -q .; then
    echo "FATAL: $d has a filesystem/partition — aborting (won't wipe data)"; exit 1; fi
done
echo ">>> all 7 drives confirmed empty & unmounted."

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
echo "Point HiCache L3 here by setting in cascade_dsr1_lite.sh:"
echo "  HICACHE_FILE_STORE_DIR=$MNT/cascade_dsr1_l3_...   (instead of /tmp/...)"
