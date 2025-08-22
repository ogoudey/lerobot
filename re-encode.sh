cd data/fine-tuning-dataset/videos/chunk-000

for cam in observation.images.front observation.images.side; do
  for f in $cam/*.mp4; do
    tmp="${f%.mp4}_h264.mp4"
    ffmpeg -y -i "$f" -c:v libx264 -pix_fmt yuv420p -crf 23 -preset veryfast "$tmp"
    mv "$tmp" "$f"
  done
done

