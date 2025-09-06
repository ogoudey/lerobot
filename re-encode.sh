#!/bin/bash
set -euo pipefail
ROOT="data/fg/videos"

find "$ROOT" -type f -name "*.mp4" | while read -r f; do
    codec=$(/usr/bin/ffprobe -v error -select_streams v:0 -show_entries stream=codec_name \
             -of default=noprint_wrappers=1:nokey=1 "$f" || echo "none")

    if [ "$codec" = "av1" ]; then
        tmp="${f%.mp4}_h264.mp4"
        echo "Re-encoding '$f'"
        if /usr/bin/ffmpeg -y -i "$f" -c:v libx264 -pix_fmt yuv420p -crf 23 -preset veryfast -an "$tmp"; then
            mv "$tmp" "$f"
            echo " ✅ Replaced '$f' with H.264"
        else
            echo "❌ Failed: $f"
            rm -f "$tmp"
        fi
    else
        echo "Skipping $f ($codec)"
    fi
done
