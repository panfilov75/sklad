"""Озвучка диктором (Piper, голос ru-irinia-medium) -> audio/vo/NN.wav + длительности."""
import json, subprocess, sys, wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = sys.argv[1]  # путь к ru-irinia-medium.onnx
LINES = json.loads((ROOT / "src/script.json").read_text())["voice"]
out = {}
for i, line in enumerate(LINES):
    wav = ROOT / f"audio/vo/{i:02d}.wav"
    subprocess.run(["piper", "--model", MODEL, "--output_file", str(wav),
                    "--length_scale", str(line.get("speed", 0.9)), "--sentence_silence", "0"],
                   input=line["text"].encode(), check=True, capture_output=True)
    with wave.open(str(wav)) as w:
        out[i] = round(w.getnframes() / w.getframerate(), 3)
    print(i, out[i], line["text"])
(ROOT / "audio/vo/durations.json").write_text(json.dumps(out, indent=1))
