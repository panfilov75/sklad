"""Вырезает иконку и надпись «Лисапед» из logo.jpg кеингом по оранжевому фону."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
img = np.asarray(Image.open(ROOT / "assets/src/logo.jpg").convert("RGB")).astype(np.float32)
OUT = ROOT / "assets/cut"


def bg_distance(a):
    # фон — плавный оранжевый градиент: оцениваем его по строкам (крайние столбцы)
    edge = np.concatenate([a[:, :40], a[:, -40:]], axis=1).mean(axis=1, keepdims=True)
    return np.sqrt(((a - edge) ** 2).sum(-1))


def save(alpha, box, name):
    l, t, r, b = box
    rgba = np.dstack([img[t:b, l:r], alpha[t:b, l:r] * 255]).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(rgba, "RGBA")
    im.crop(im.getbbox()).save(OUT / f"{name}.png")


d = bg_distance(img)

# иконка: белый пакет + листья + багет; колесо замыкаем морфологией и заливкой дыр
box = (250, 180, 760, 660)
# белое (пакет), зелёное (листья), красно-оранжевое (велосипед/обод колеса),
# багет — светлее фона, ищем только в его зоне; тень пакета не попадает
r, g = img[..., 0], img[..., 1]
core = (img.min(-1) > 200) | (g > r) | ((r > 225) & (g < 70))
yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
loaf_zone = (yy < 300) & (xx > 465) & (xx < 625)
core |= loaf_zone & (g > 128)
core[:, : box[0]] = core[:, box[2]:] = False
core[: box[1]] = core[box[3]:] = False
core = ndi.binary_opening(core, iterations=2)
solid = ndi.binary_fill_holes(ndi.binary_closing(core, structure=np.ones((3, 3)), iterations=6))
lab, n = ndi.label(solid)
sizes = ndi.sum(solid, lab, range(1, n + 1))
solid = lab == (np.argmax(sizes) + 1)
alpha = ndi.gaussian_filter(solid.astype(np.float32), 1.2)
save(alpha, box, "logo_icon")

# надпись: белый текст -> альфа по «белизне»
box = (120, 650, 900, 860)
mn = img.min(-1)
alpha = np.clip((mn - 60) / 150, 0, 1)
alpha[: box[1]] = 0
white = np.full_like(img, 255)
img = np.where(alpha[..., None] > 0, white, img)
save(alpha, box, "logo_word")
print("ok")
