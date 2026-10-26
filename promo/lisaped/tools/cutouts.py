"""Использование: python3 tools/cutouts.py [имя ...]
Вырезает элементы из исходных макетов (фон убирается rembg) -> assets/cut/*.png"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from rembg import remove, new_session

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "assets/src", ROOT / "assets/cut"
OUT.mkdir(parents=True, exist_ok=True)

# имя: (исходник, кроп (l, t, r, b) или None)
JOBS = {
    "backpack": ("backpack.png", None),
    "phone_app": ("poster_white.png", (95, 480, 445, 1150)),
    "groceries": ("poster_white.png", (400, 620, 1055, 1160)),
    "hand_phone": ("banner_promo.jpg", (700, 140, 1060, 833)),  # isnet: телефон без руки
    "brands": ("banner_promo.jpg", (0, 600, 1179, 808)),
    "bag_orange": ("poster_orange.png", (420, 580, 941, 1460)),
}

# эти лучше режет birefnet-general (тяжёлая модель: по одному процессу на картинку)
BIREFNET = {"groceries", "brands", "bag_orange"}
SOLID = {"phone_app", "hand_phone"}

names = sys.argv[1:] or list(JOBS)
for name in names:
    src, box = JOBS[name]
    session = new_session("birefnet-general" if name in BIREFNET else "isnet-general-use")
    img = Image.open(SRC / src).convert("RGB")
    if box:
        img = img.crop(box)
    cut = remove(img, session=session, post_process_mask=True)
    if name in SOLID:
        # белый экран телефона rembg частично «съедает» — заливаем дыры в маске
        a = np.asarray(cut)[..., 3] > 128
        a = ndi.binary_fill_holes(ndi.binary_closing(a, iterations=8))
        lab, n = ndi.label(a)  # оставляем только сам телефон
        a = lab == (np.argmax(ndi.sum(a, lab, range(1, n + 1))) + 1)
        alpha = (ndi.gaussian_filter(a.astype(np.float32), 0.8) * 255).astype(np.uint8)
        cut = img.convert("RGBA")
        cut.putalpha(Image.fromarray(alpha))
        cut = cut.crop(cut.getbbox())
    cut = cut.crop(cut.getbbox())
    cut.save(OUT / f"{name}.png")
    print(name, cut.size)

# значки сторов — просто кропы с белого постера
src = Image.open(SRC / "poster_white.png").convert("RGB")
src.crop((720, 1298, 942, 1370)).save(OUT / "badge_gplay.png")
src.crop((720, 1380, 942, 1452)).save(OUT / "badge_appstore.png")
