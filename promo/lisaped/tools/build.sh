#!/usr/bin/env bash
# Полная сборка ролика: озвучка -> звук -> кадры -> mp4.
# Требуется: piper-tts + модель ru-irinia-medium.onnx, ffmpeg, node + playwright, python3 (numpy, scipy).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${PIPER_MODEL:?укажите PIPER_MODEL=/путь/к/ru-irinia-medium.onnx}
python3 tools/voice.py "$MODEL"
python3 tools/audio.py
node tools/render.js video
ffmpeg -y -loglevel error -i build/frames.mp4 -i build/audio.wav -c:v copy \
  -af loudnorm=I=-14:TP=-1.5:LRA=9 -c:a aac -b:a 192k -ar 44100 -movflags +faststart -shortest \
  lisaped_promo_1080x1920.mp4
echo "готово: lisaped_promo_1080x1920.mp4"
