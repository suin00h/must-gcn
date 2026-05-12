#!/usr/bin/env bash
# Download HRNet-inferenced 2D pose annotations for NTU-RGBD from PYSKL.
# Format: COCO-17 keypoints (not NTU's native 25 joints).
# See: https://github.com/kennymckormick/pyskl/tree/main/data
#
# Citation (PYSKL / PoseConv3D):
#   Duan et al. "Revisiting Skeleton-based Action Recognition", CVPR 2022.

set -e

# Default to <project>/data/ntu-rgbd/processed/hrnet.  Override with
#   DATA_ROOT=/path/to/data  bash data_prep/download_hrnet.sh
# or
#   OUT_DIR=/path/to/dir     bash data_prep/download_hrnet.sh
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/ntu-rgbd/processed/hrnet}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

BASE="https://download.openmmlab.com/mmaction/pyskl/data/nturgbd"

FILES=(
    "ntu60_hrnet.pkl"
    "ntu120_hrnet.pkl"
    # 3D Kinect annotations (uncomment if needed):
    # "ntu60_3danno.pkl"
    # "ntu120_3danno.pkl"
)

for f in "${FILES[@]}"; do
    if [[ -f "$f" ]]; then
        echo "[skip] $f already exists ($(du -h "$f" | cut -f1))"
        continue
    fi
    echo "[get ] $f"
    wget -c -q --show-progress "$BASE/$f"
done

echo
echo "Done. Files in $OUT_DIR:"
ls -lh "$OUT_DIR"
