
import cv2
import numpy as np


def _warp_triangle_to_overlay_premult(
    src_bgra,
    out_rgb_premult,
    out_alpha,
    t_src,
    t_dst,
    pad=3,
    feather=1.6
):
    """
    Warp triangle BGRA -> overlay prémultiplié (RGB_premult + alpha).
    Objectif: supprimer les micro-coutures en compositant d'abord dans un overlay,
    puis en blend une seule fois sur la frame.
    """
    t_src = np.float32(t_src)
    t_dst = np.float32(t_dst)

    Hs, Ws = src_bgra.shape[:2]
    Hd, Wd = out_alpha.shape[:2]

    r1 = cv2.boundingRect(t_src)
    r2 = cv2.boundingRect(t_dst)

    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return

    # padding ROI
    x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
    x1e = min(Ws, x1 + w1 + pad); y1e = min(Hs, y1 + h1 + pad)

    x2p = max(0, x2 - pad); y2p = max(0, y2 - pad)
    x2e = min(Wd, x2 + w2 + pad); y2e = min(Hd, y2 + h2 + pad)

    w1p, h1p = x1e - x1p, y1e - y1p
    w2p, h2p = x2e - x2p, y2e - y2p
    if w1p <= 0 or h1p <= 0 or w2p <= 0 or h2p <= 0:
        return

    src_roi = src_bgra[y1p:y1e, x1p:x1e]
    if src_roi.size == 0:
        return

    t1 = t_src - np.array([x1p, y1p], dtype=np.float32)
    t2 = t_dst - np.array([x2p, y2p], dtype=np.float32)

    M = cv2.getAffineTransform(t1, t2)

    # --- Premult du ROI source ---
    src_f = src_roi.astype(np.float32) / 255.0
    src_rgb = src_f[:, :, :3]
    src_a = src_f[:, :, 3:4]  # (h,w,1)
    src_rgb_p = src_rgb * src_a

    # warp RGB premult + warp alpha
    warp_rgb_p = cv2.warpAffine(
        src_rgb_p, M, (w2p, h2p),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    warp_a = cv2.warpAffine(
        src_a, M, (w2p, h2p),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    if warp_a.ndim == 2:
        warp_a = warp_a[:, :, None]

    # masque triangle + feather
    tri_mask = np.zeros((h2p, w2p), dtype=np.float32)
    cv2.fillConvexPoly(tri_mask, np.int32(t2), 1.0, lineType=cv2.LINE_AA)
    if feather and feather > 0:
        tri_mask = cv2.GaussianBlur(tri_mask, (0, 0), feather)
    tri_mask = tri_mask[:, :, None]

    srcA = np.clip(warp_a * tri_mask, 0.0, 1.0)             # (h,w,1)
    srcRGBp = np.clip(warp_rgb_p * tri_mask, 0.0, 1.0)      # (h,w,3) premult

    # ROI destination overlay
    dstA = out_alpha[y2p:y2e, x2p:x2e][:, :, None]           # (h,w,1)
    dstRGBp = out_rgb_premult[y2p:y2e, x2p:x2e]              # (h,w,3)

    # "Over" en prémultiplié: out = src + dst*(1-srcA)
    one_minus = (1.0 - srcA)
    out_rgb_p = srcRGBp + dstRGBp * one_minus
    out_a = srcA + dstA * one_minus

    out_rgb_premult[y2p:y2e, x2p:x2e] = out_rgb_p
    out_alpha[y2p:y2e, x2p:x2e] = out_a[:, :, 0]


def render_face_overlay(mask_bgra, src_points, dst_points, triangles, out_w, out_h, pad=3, feather=1.6):
    """
    Rend le masque (BGRA) en overlay BGRA (taille frame) via triangles,
    en accumulation prémultipliée (anti micro-coutures).
    """
    out_rgb_p = np.zeros((out_h, out_w, 3), dtype=np.float32)
    out_a = np.zeros((out_h, out_w), dtype=np.float32)

    for (i, j, k) in triangles:
        _warp_triangle_to_overlay_premult(
            mask_bgra, out_rgb_p, out_a,
            [src_points[i], src_points[j], src_points[k]],
            [dst_points[i], dst_points[j], dst_points[k]],
            pad=pad, feather=feather
        )

    # Unpremultiply
    eps = 1e-6
    a = np.clip(out_a, 0.0, 1.0)
    rgb = out_rgb_p / (a[:, :, None] + eps)
    rgb = np.clip(rgb, 0.0, 1.0)

    overlay = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    overlay[:, :, :3] = (rgb * 255).astype(np.uint8)
    overlay[:, :, 3] = (a * 255).astype(np.uint8)
    return overlay


def warp_mask_to_face(mask_rgba, frame_bgr, src_points, dst_points, triangles, pad=3, feather=1.6):
    """
    Compat: conserve le même nom, mais rend via overlay puis blend.
    """
    H, W = frame_bgr.shape[:2]
    overlay = render_face_overlay(mask_rgba, src_points, dst_points, triangles, W, H, pad=pad, feather=feather)

    # Blend plein écran (alpha over)
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    fg = overlay[:, :, :3].astype(np.float32)
    bg = frame_bgr.astype(np.float32)
    out = bg * (1.0 - alpha) + fg * alpha
    frame_bgr[:, :, :] = np.clip(out, 0, 255).astype(np.uint8)
