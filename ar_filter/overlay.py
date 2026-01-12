
import cv2
import numpy as np


def load_rgba(path: str) -> np.ndarray:
    """
    Charge une image en conservant les canaux (IMREAD_UNCHANGED).
    - BGRA (4 canaux) -> OK
    - BGR (3 canaux) -> alpha=255
    - GRAY (1 canal) -> BGR puis alpha=255
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Impossible de lire: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 3:
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)

    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"{path} doit être convertible en BGRA (4 canaux). Shape={img.shape}")

    return img


def alpha_blend_at(bg_bgr: np.ndarray, fg_bgra: np.ndarray, x: int, y: int) -> np.ndarray:
    H, W = bg_bgr.shape[:2]
    h, w = fg_bgra.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x1 >= x2 or y1 >= y2:
        return bg_bgr

    fg = fg_bgra[y1 - y:y2 - y, x1 - x:x2 - x]
    bg = bg_bgr[y1:y2, x1:x2]

    alpha = fg[:, :, 3:4].astype(np.float32) / 255.0
    fg_rgb = fg[:, :, :3].astype(np.float32)
    bg_f = bg.astype(np.float32)

    out = bg_f * (1.0 - alpha) + fg_rgb * alpha
    bg_bgr[y1:y2, x1:x2] = out.astype(np.uint8)
    return bg_bgr


def alpha_blend_full(bg_bgr: np.ndarray, fg_bgra: np.ndarray) -> np.ndarray:
    if fg_bgra.shape[:2] != bg_bgr.shape[:2]:
        raise ValueError("alpha_blend_full: fg et bg doivent avoir la même taille.")
    alpha = fg_bgra[:, :, 3:4].astype(np.float32) / 255.0
    fg_rgb = fg_bgra[:, :, :3].astype(np.float32)
    bg_f = bg_bgr.astype(np.float32)
    out = bg_f * (1.0 - alpha) + fg_rgb * alpha
    return out.astype(np.uint8)


# Compatibilité anciens noms
def load_mask_rgba(path: str) -> np.ndarray:
    return load_rgba(path)


def alpha_blend(bg_bgr: np.ndarray, fg_bgra: np.ndarray, x: int = 0, y: int = 0) -> np.ndarray:
    if x == 0 and y == 0 and fg_bgra.shape[:2] == bg_bgr.shape[:2]:
        return alpha_blend_full(bg_bgr, fg_bgra)
    return alpha_blend_at(bg_bgr, fg_bgra, x, y)
