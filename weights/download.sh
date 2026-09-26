#!/usr/bin/env sh
# Fetch the detector weights (only needed if weights/*.pt are not already in the checkout).
# Run once, with internet, before the offline evaluation.
set -e
cd "$(dirname "$0")"
for m in yolo11m.pt; do
  [ -f "$m" ] || curl -fL -o "$m" "https://github.com/ultralytics/assets/releases/download/v8.3.0/$m"
done
# sha256 of the files we used (verify integrity)
sha256sum -c SHA256SUMS
