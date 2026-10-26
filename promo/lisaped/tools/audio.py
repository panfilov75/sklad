"""Музыка (синтез), звуковые эффекты и сведение с озвучкой -> build/audio.wav

Тайминги совпадают с src/video.html (120 BPM: такт = 2 с, дроп на 6.0 с).
"""
import json
import wave
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
SR = 44100
DUR = 28.0
N = int(SR * DUR)
BEAT = 0.5
rng = np.random.default_rng(42)


def t_arr(d):
    return np.arange(int(SR * d)) / SR


def env(d, a=0.005, decay=0.2):
    t = t_arr(d)
    return np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / decay)


def place(buf, x, at, gain=1.0):
    i = int(at * SR)
    if i >= len(buf):
        return
    x = x[: len(buf) - i]
    buf[i:i + len(x)] += x * gain


def lp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "high", fs=SR, output="sos"), x)


def bp(x, lo, hi, order=2):
    return signal.sosfilt(signal.butter(order, [lo, hi], "band", fs=SR, output="sos"), x)


def saw(freq, d, detune=(0,)):
    t = t_arr(d)
    out = np.zeros_like(t)
    for c in detune:
        f = freq * 2 ** (c / 1200)
        out += 2 * ((t * f + rng.random()) % 1) - 1
    return out / len(detune)


def midi(n):
    return 440 * 2 ** ((n - 69) / 12)


def reverb(x, length=1.2, mix=0.25):
    ir = rng.standard_normal(int(SR * length)) * np.exp(-t_arr(length) / (length / 5))
    ir = lp(ir, 6000)
    wet = signal.fftconvolve(x, ir)[: len(x)]
    wet *= np.abs(x).max() / (np.abs(wet).max() + 1e-9)
    return x * (1 - mix) + wet * mix


# ---------- инструменты ----------
def kick():
    t = t_arr(0.45)
    f = 45 + 110 * np.exp(-t / 0.04)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18)
    click = hp(rng.standard_normal(len(t)), 2000) * np.exp(-t / 0.004) * 0.3
    return np.tanh((body + click) * 1.6)


def clap():
    d = 0.3
    n = rng.standard_normal(int(SR * d))
    e = np.zeros(len(n))
    for k, off in enumerate((0, 0.011, 0.022)):
        i = int(off * SR)
        e[i:] += np.exp(-np.arange(len(n) - i) / SR / (0.012 if k < 2 else 0.12))
    return bp(n, 900, 5000) * e


def hat(open_=False):
    d = 0.25 if open_ else 0.06
    return hp(rng.standard_normal(int(SR * d)), 7000) * env(d, 0.001, 0.09 if open_ else 0.018)


def bass_note(n, d):
    x = saw(midi(n), d, (-6, 6)) * 0.6 + np.sin(2 * np.pi * midi(n) * t_arr(d)) * 0.7
    return lp(x, 700) * env(d, 0.004, d * 0.7)


def stab(notes, d, cutoff=3200, decay=0.16):
    x = sum(saw(midi(n), d, (-14, -5, 0, 5, 14)) for n in notes) / len(notes)
    return lp(x, cutoff) * env(d, 0.003, decay)


def pluck(n, d=0.25):
    t = t_arr(d)
    f = midi(n)
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * env(d, 0.002, 0.09)


def pad(notes, d):
    x = sum(saw(midi(n), d, (-10, 0, 10)) for n in notes) / len(notes)
    e = np.minimum(1, t_arr(d) / 0.4) * np.minimum(1, (d - t_arr(d)) / 0.4)
    return lp(x, 900) * e


# аккорды по тактам (I–V–vi–IV в до мажоре); корни для баса
PROG = [([60, 64, 67], 36), ([59, 62, 67], 43), ([57, 60, 64], 45), ([57, 60, 65], 41)]


def music():
    drums, bass, synth = np.zeros(N), np.zeros(N), np.zeros(N)
    K, C = kick(), clap()

    def bar_groove(t0, full=True, stabs=True):
        chord, root = PROG[int(round(t0 / 2)) % 4]
        for b in range(4):
            tb = t0 + b * BEAT
            place(drums, K, tb, 1.0)
            if full and b % 2 == 1:
                place(drums, C, tb, 0.55)
            place(drums, hat(open_=True), tb + BEAT / 2, 0.22)
            for s in (0.25, 0.75):
                place(drums, hat(), tb + s * BEAT, 0.12)
            place(bass, bass_note(root, 0.22), tb + BEAT / 2, 0.5)
        if stabs:
            for s16 in (0, 3, 6, 10, 12):
                place(synth, stab([n + 12 for n in chord], 0.3), t0 + s16 * BEAT / 4, 0.32)
        arp = chord + [chord[0] + 12]
        for i in range(16):
            place(synth, pluck(arp[i % 4] + 24), t0 + i * BEAT / 4, 0.08)

    # 0–3: хук — бит без стабов, на 2.0 полноценный такт
    bar_groove(0.0, full=True, stabs=False)
    bar_groove(2.0, full=True, stabs=True)
    # 3–6: «грустный» провал — приглушённый пэд
    place(synth, pad([57, 60, 64], 1.5), 3.0, 0.35)
    place(synth, pad([53, 57, 60], 1.6), 4.4, 0.35)
    # дроп 6–26
    for t0 in np.arange(6.0, 26.0, 2.0):
        bar_groove(t0, full=True, stabs=True)
    # финальный аккорд
    place(drums, K, 26.0, 1.0)
    place(synth, stab([48, 60, 64, 67, 72], 2.0, 4000, decay=0.9), 26.0, 0.35)
    place(bass, bass_note(36, 1.6), 26.0, 0.6)

    # провал после хука: обрезаем хвосты бита 3.0–6.0
    i0, i1 = int(3.0 * SR), int(6.0 * SR)
    gate = np.zeros(i1 - i0)
    gate[: int(0.08 * SR)] = np.linspace(1, 0, int(0.08 * SR))
    for b in (drums, bass):
        b[i0:i1] *= gate
    synth[i0:i0 + int(0.06 * SR)] *= 0.2
    synth[i0 + int(0.06 * SR):i1] = lp(synth[i0 + int(0.06 * SR):i1], 1200)

    synth = reverb(synth, 1.0, 0.3)
    mix = drums * 0.9 + bass * 0.9 + synth * 0.9
    return mix


# ---------- эффекты ----------
def whoosh(d=0.45, up=True):
    n = rng.standard_normal(int(SR * d))
    t = t_arr(d)
    out = np.zeros_like(n)
    # скользящий полосовой фильтр блоками
    blk = 512
    for i in range(0, len(n), blk):
        k = i / len(n)
        fc = 400 + (3500 if up else 2500) * (k if up else 1 - k)
        seg = n[max(0, i - 2048):i + blk]
        out[i:i + blk] = bp(seg, fc * 0.6, fc * 1.4)[-len(n[i:i + blk]):]
    shape = np.sin(np.pi * np.clip(t / d, 0, 1)) ** 1.5
    return out * shape * 1.4


def impact():
    t = t_arr(1.2)
    boom = np.sin(2 * np.pi * np.cumsum(30 + 80 * np.exp(-t / 0.08)) / SR) * np.exp(-t / 0.45)
    crack = lp(rng.standard_normal(len(t)), 3000) * np.exp(-t / 0.06) * 0.6
    return np.tanh((boom + crack) * 1.5)


def pop(f=700):
    t = t_arr(0.12)
    return np.sin(2 * np.pi * np.cumsum(f * (1 + 1.5 * np.exp(-t / 0.015))) / SR) * env(0.12, 0.001, 0.03)


def tick():
    return bp(rng.standard_normal(int(SR * 0.03)), 2500, 6000) * env(0.03, 0.0005, 0.005)


def ding(f=1568):
    t = t_arr(1.2)
    x = sum(a * np.sin(2 * np.pi * f * m * t) * np.exp(-t / (0.5 / m)) for m, a in ((1, 1), (2.76, .4), (5.4, .2)))
    return x * 0.5


def cash():
    x = np.zeros(int(SR * 1.3))
    place(x, ding(2093), 0.0, 0.7)
    place(x, ding(2637), 0.08, 0.6)
    place(x, hp(rng.standard_normal(int(SR * 0.25)), 3000) * env(0.25, 0.002, 0.06), 0.0, 0.4)
    return x


def riser(d):
    t = t_arr(d)
    k = t / d
    f = 200 * 2 ** (k * 3)
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.3
    noise = hp(rng.standard_normal(len(t)), 1500) * 0.5
    return (tone + noise) * k ** 2


def snare_roll(t0, t1, buf):
    t = t0
    while t < t1:
        k = (t - t0) / (t1 - t0)
        place(buf, bp(rng.standard_normal(int(SR * 0.1)), 300, 6000) * env(0.1, 0.001, 0.04), t, 0.12 + 0.35 * k)
        t += BEAT / 2 if k < 0.5 else (BEAT / 4 if k < 0.8 else BEAT / 8)


def tape_stop():
    t = t_arr(0.5)
    f = 220 * np.exp(-t / 0.15)
    return lp(np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)), 1500) * (1 - t / 0.5) * 0.5


def fly(d):
    t = t_arr(d)
    f = 210 + 25 * np.sin(2 * np.pi * 7 * t) + 15 * np.sin(2 * np.pi * 1.3 * t)
    x = bp(2 * ((np.cumsum(f) / SR) % 1) - 1, 150, 2500)
    return x * (0.6 + 0.4 * np.sin(2 * np.pi * 0.9 * t)) * np.minimum(1, t / 0.2) * np.minimum(1, (d - t) / 0.2)


def creak():
    t = t_arr(0.5)
    return bp(rng.standard_normal(len(t)), 150, 600) * np.exp(-t / 0.15) * 1.2


def sfx():
    b = np.zeros(N)
    W, Wd = whoosh(0.45, True), whoosh(0.4, False)
    I, P = impact(), pop()
    place(b, I, 0.0, 0.6)
    place(b, W, 0.0, 0.5)
    place(b, pop(500), 0.45, 0.6)
    place(b, I, 1.38, 0.7)
    place(b, pop(900), 1.85, 0.5)
    place(b, Wd, 2.6, 0.6)
    place(b, tape_stop(), 2.95, 0.6)
    place(b, creak(), 3.45, 0.5)
    place(b, fly(1.8), 4.0, 0.07)
    place(b, pop(600), 4.45, 0.45)
    place(b, riser(1.5), 4.5, 0.5)
    snare_roll(5.0, 6.0, b)
    place(b, I, 6.0, 1.0)
    place(b, ding(2349), 6.05, 0.25)
    place(b, W, 6.25, 0.35)
    place(b, whoosh(0.5, True), 7.4, 0.8)
    place(b, P, 7.95, 0.5)
    place(b, Wd, 8.85, 0.5)
    t = 9.0
    while t < 9.9:
        place(b, tick(), t, 0.7)
        t += 0.11
    place(b, I, 9.25, 0.4)
    place(b, ding(1976), 9.9, 0.4)
    place(b, W, 10.4, 0.5)
    for t0 in (10.5, 11.3, 11.83, 12.75):
        place(b, pop(650 + 80 * t0 % 300), t0, 0.5)
    place(b, W, 13.6, 0.45)
    place(b, I, 13.7, 0.55)
    place(b, W, 16.0, 0.5)
    for t0 in (16.75, 17.35):
        place(b, tick(), t0, 1.0)
        place(b, pop(1100), t0 + 0.06, 0.5)
    place(b, W, 18.6, 0.5)
    place(b, I, 18.6, 0.35)
    place(b, pop(800), 20.1, 0.5)
    place(b, I, 20.62, 0.7)
    place(b, cash(), 20.9, 0.5)
    place(b, impact()[: int(0.4 * SR)], 21.55, 0.5)
    place(b, Wd, 23.0, 0.5)
    for t0 in (23.35, 24.55, 24.7, 25.2):
        place(b, P, t0, 0.45)
    place(b, ding(2093), 26.0, 0.25)
    return b


# ---------- озвучка ----------
VO_AT = [0.15, 2.95, 6.25, 10.5, 16.1, 18.7, 23.1]


def load_vo(path):
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
    x = signal.resample_poly(x, SR, sr)
    x = hp(x, 90)
    x = x / (np.abs(x).max() + 1e-9)
    # мягкая компрессия: выравниваем громкость слогов
    return np.tanh(x * 2.2) / np.tanh(2.2)


def main():
    vo = np.zeros(N)
    for i, at in enumerate(VO_AT):
        place(vo, load_vo(ROOT / f"audio/vo/{i:02d}.wav"), at)

    m = music()
    # дакинг музыки под голос
    e = np.abs(vo)
    e = signal.sosfilt(signal.butter(1, 4, "low", fs=SR, output="sos"), e)
    e = np.clip(e / (e.max() + 1e-9) * 3, 0, 1)
    m *= 1 - 0.55 * e

    fx = sfx()
    mix = m * 0.42 + fx * 0.45 + vo * 0.95
    # затухание в конце
    fade = int(0.6 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)
    mix = np.tanh(mix * 1.1) / 1.1
    mix /= np.abs(mix).max() / 0.95

    # лёгкая стереобаза: музыку/эффекты чуть расширяем задержкой
    wide = np.roll(m * 0.42 + fx * 0.45, int(0.012 * SR)) * 0.15
    left, right = mix + wide, mix - wide
    st = np.stack([left, right], 1)
    st /= np.abs(st).max() / 0.95
    out = ROOT / "build/audio.wav"
    out.parent.mkdir(exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())
    print("ok", out)


if __name__ == "__main__":
    main()
