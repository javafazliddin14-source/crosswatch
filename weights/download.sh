#!/usr/bin/env sh
# Fetch the detector weights (only needed if weights/*.pt are not already in the checkout).
# Run once, with internet, before the offline evaluation.
# yolo11m.pt is the submission's detector; yolo11n.pt is only used to build the browser demo model.
set -e
cd "$(dirname "$0")"
for m in yolo11m.pt yolo11n.pt; do
  [ -f "$m" ] || curl -fL -o "$m" "https://github.com/ultralytics/assets/releases/download/v8.3.0/$m"
done
# sha256 of the files we used (verify integrity)
sha256sum -c SHA256SUMS
