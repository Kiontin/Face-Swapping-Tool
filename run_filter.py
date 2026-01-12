
import argparse
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ar_filter.overlay import load_mask_rgba, alpha_blend
from ar_filter.tracker import FaceLandmarkTracker
from ar_filter.warp import warp_mask_to_face
from ar_filter.hair_segmenter import AsyncHairMask

FACE_MODEL = "assets/face_landmarker.task"
SEG_MODEL = "assets/selfie_multiclass_256x256.tflite"

MASK_PNG = "assets/mask.png"
WIG_PNG = "assets/wig.png"
WIG_POINTS = "assets/wig_src_points.npy"

# --------- Réglages qualité ---------
WARP_PAD = 3
WARP_FEATHER = 1.9

HAIR_DILATE = 10
HAIR_BLUR = 17
HAIR_FEATHER_PX = 18

# Boost haut de tête (maintenant elliptique)
TOP_BOOST = 0.55        # quantité de boost max
TOP_BAND = 0.30         # hauteur relative au-dessus du front (en % de hauteur visage)
TOP_MIN = 0.65          # plancher de masque cheveux dans la zone boost
TOP_FEATHER = 19        # flou du masque elliptique (bords doux)

FACE_REGION_EXPAND = 22
FACE_REGION_BLUR = 31


# --------- OneEuro pour landmarks (stable + réactif) ---------
def _smoothing_factor(te, cutoff):
    r = 2.0 * np.pi * cutoff * te
    return r / (r + 1.0)

def _exponential_smoothing(a, x, x_prev):
    return a * x + (1.0 - a) * x_prev

class OneEuroFilter:
    def __init__(self, min_cutoff=1.6, beta=0.018, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t=None):
        if t is None:
            t = time.time()
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x.astype(np.float32)
            self.dx_prev = np.zeros_like(self.x_prev, dtype=np.float32)
            return self.x_prev

        te = max(1e-6, float(t - self.t_prev))
        self.t_prev = t

        x = x.astype(np.float32)
        dx = (x - self.x_prev) / te

        a_d = _smoothing_factor(te, self.d_cutoff)
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)
        self.dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * np.mean(np.abs(dx_hat))
        a = _smoothing_factor(te, cutoff)
        x_hat = _exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        return x_hat


def warp_rgba_affine(src_rgba, M, out_w, out_h):
    return cv2.warpAffine(
        src_rgba, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )


def face_region_mask_from_landmarks(dst_points, W, H, expand_px=22, blur=31, kernel=None):
    hull = cv2.convexHull(dst_points.astype(np.float32))
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(m, hull.astype(np.int32), 255, lineType=cv2.LINE_AA)

    if expand_px > 0:
        if kernel is None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expand_px + 1, 2 * expand_px + 1))
        m = cv2.dilate(m, kernel, iterations=1)

    blur = max(3, int(blur) | 1)
    m = cv2.GaussianBlur(m, (blur, blur), 0)
    return m.astype(np.float32) / 255.0


def apply_alpha_mask(bgra, mask01):
    out = bgra.copy()
    a = out[:, :, 3].astype(np.float32) / 255.0
    a = a * mask01
    out[:, :, 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return out


def edge_pad_rgba_inpaint(bgra):
    out = bgra.copy()
    alpha = out[:, :, 3]
    mask = (alpha == 0).astype(np.uint8) * 255
    if mask.sum() > 0:
        bgr = out[:, :, :3]
        out[:, :, :3] = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    return out


def refine_hair_mask(hm_conf01, dilate_px=10, blur=17, feather_px=18):
    hm = np.clip(hm_conf01, 0.0, 1.0).astype(np.float32)
    if hm.ndim == 3 and hm.shape[2] == 1:
        hm = hm[:, :, 0]

    core = (hm > 0.45).astype(np.uint8) * 255

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        core = cv2.dilate(core, k, iterations=1)

    inside = cv2.distanceTransform((core > 0).astype(np.uint8), cv2.DIST_L2, 3)
    inside = np.clip(inside / max(1.0, float(feather_px)), 0.0, 1.0)

    outside = cv2.distanceTransform((core == 0).astype(np.uint8), cv2.DIST_L2, 3)
    outside = np.clip(outside / max(1.0, float(feather_px)), 0.0, 1.0)
    outside = 1.0 - outside

    feather = np.clip(inside * outside, 0.0, 1.0)
    mask = np.maximum(feather, hm * 0.90)

    blur = max(3, int(blur) | 1)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return np.clip(mask, 0.0, 1.0)


def boost_top_of_head_elliptical(hm01, dst_points, W, H):
    """
    ✅ Boost non-rectangulaire:
    - crée une ellipse au-dessus du front (entre tempes)
    - applique un gradient (plus fort en haut) + feather doux
    => plus de "rectangle" visible.
    """
    hm = hm01  # modif in-place
    try:
        # Landmarks clés
        y_fore = float(dst_points[10][1])     # haut du front
        y_chin = float(dst_points[152][1])    # menton
        x_l = float(min(dst_points[234][0], dst_points[454][0]))
        x_r = float(max(dst_points[234][0], dst_points[454][0]))

        face_h = max(1.0, y_chin - y_fore)
        band_h = TOP_BAND * face_h

        # Zone verticale de boost
        y1 = int(max(0, y_fore - band_h))
        y2 = int(max(0, y_fore))
        if y2 <= y1:
            return hm

        # Centre de l'ellipse (un peu au-dessus du front)
        cx = int(np.clip((x_l + x_r) * 0.5, 0, W - 1))
        cy = int(np.clip(y1 + (y2 - y1) * 0.45, 0, H - 1))

        # Axes: largeur ~ tempes, hauteur ~ band_h
        ax = int(max(10, (x_r - x_l) * 0.60))
        ay = int(max(10, (y2 - y1) * 0.75))

        # Masque ellipse
        ell = np.zeros((H, W), dtype=np.float32)
        cv2.ellipse(ell, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1, lineType=cv2.LINE_AA)

        # Feather doux sur l'ellipse pour éviter toute forme dure
        k = max(3, int(TOP_FEATHER) | 1)
        ell = cv2.GaussianBlur(ell, (k, k), 0)
        ell = np.clip(ell, 0.0, 1.0)

        # Gradient vertical (plus fort en haut)
        g = np.zeros((H, 1), dtype=np.float32)
        g[y1:y2, 0] = np.linspace(1.0, 0.0, y2 - y1, dtype=np.float32)
        g = g[:, 0]  # (H,)
        grad = ell * g[:, None]

        # Appliquer plancher + boost progressif
        hm = np.maximum(hm, TOP_MIN * grad)
        hm = np.clip(hm + TOP_BOOST * grad, 0.0, 1.0)

    except Exception:
        pass
    return hm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cv2.setUseOptimized(True)
    cv2.setNumThreads(1)

    mask_rgba = load_mask_rgba(MASK_PNG)
    wig = edge_pad_rgba_inpaint(load_mask_rgba(WIG_PNG))
    wig_src_points = np.load(WIG_POINTS).astype(np.float32)

    tracker = FaceLandmarkTracker(model_path=FACE_MODEL, use_gpu=False)
    hair_async = AsyncHairMask(SEG_MODEL, use_gpu=True, min_cutoff=1.0, beta=0.03, d_cutoff=1.0)
    lm_filter = OneEuroFilter(min_cutoff=1.6, beta=0.018, d_cutoff=1.0)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    kernel_face = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * FACE_REGION_EXPAND + 1, 2 * FACE_REGION_EXPAND + 1))

    # Préparation src_points/triangles via FaceLandmarker (IMAGE) sur mask.png
    def compute_src_points_and_tris(mask_rgba_bgra):
        rgb = cv2.cvtColor(mask_rgba_bgra[:, :, :3], cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        H, W = rgb.shape[:2]

        base_options = python.BaseOptions(model_asset_path=FACE_MODEL)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        res = landmarker.detect(mp_image)
        landmarker.close()

        if not res.face_landmarks:
            raise RuntimeError("Aucun visage détecté sur mask.png")

        lms = res.face_landmarks[0]
        src_pts = np.array([(lm.x * W, lm.y * H) for lm in lms], dtype=np.float32)

        subdiv = cv2.Subdiv2D((0, 0, W, H))
        for (x, y) in src_pts:
            subdiv.insert((float(x), float(y)))
        tri_list = subdiv.getTriangleList().reshape(-1, 3, 2)

        pts = src_pts.astype(np.float32)
        tris_idx = []
        for tri in tri_list:
            idx = []
            ok = True
            for (x, y) in tri:
                d = np.sum((pts - np.array([x, y], dtype=np.float32)) ** 2, axis=1)
                i = int(np.argmin(d))
                if d[i] > 3.0**2:
                    ok = False
                    break
                idx.append(i)
            if ok and len(set(idx)) == 3:
                tris_idx.append(tuple(idx))
        tris_idx = list(dict.fromkeys(tris_idx))
        return src_pts, np.array(tris_idx, dtype=np.int32)

    src_points, triangles = compute_src_points_and_tris(mask_rgba)

    idx_affine_full = [33, 263, 10, 152, 234, 454]
    idx_affine_wig = [33, 263, 1, 152]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            H, W = frame.shape[:2]
            frame_in = frame.copy()

            # hair mask async
            hair_async.submit(frame_in)
            hm_conf = hair_async.get_mask()
            if hm_conf is None:
                hm_conf = np.zeros((H, W), dtype=np.float32)
            if hm_conf.ndim == 3 and hm_conf.shape[2] == 1:
                hm_conf = hm_conf[:, :, 0]

            # landmarks live
            dst_points = tracker.get_landmarks_points(frame_in)
            if dst_points is None:
                cv2.imshow("Masque + wig (q/ESC)", frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
                continue

            dst_points = lm_filter(dst_points, t=time.time())

            # hair refine + boost ovale (✅ plus rectangulaire)
            hm = refine_hair_mask(hm_conf, dilate_px=HAIR_DILATE, blur=HAIR_BLUR, feather_px=HAIR_FEATHER_PX)
            hm = boost_top_of_head_elliptical(hm, dst_points, W, H)

            # A) affine globale masque
            if src_points.shape[0] > max(idx_affine_full) and dst_points.shape[0] > max(idx_affine_full):
                srcA = src_points[idx_affine_full].astype(np.float32)
                dstA = dst_points[idx_affine_full].astype(np.float32)
                M_full, _ = cv2.estimateAffinePartial2D(srcA, dstA, method=cv2.LMEDS)
                if M_full is not None:
                    mask_full = warp_rgba_affine(mask_rgba, M_full, W, H)
                    face_region = face_region_mask_from_landmarks(dst_points, W, H, expand_px=FACE_REGION_EXPAND, blur=FACE_REGION_BLUR, kernel=kernel_face)
                    mask_full = apply_alpha_mask(mask_full, face_region)
                    frame = alpha_blend(frame, mask_full, 0, 0)

            # B) mesh warp
            warp_mask_to_face(mask_rgba, frame, src_points, dst_points, triangles, pad=WARP_PAD, feather=WARP_FEATHER)

            # C) wig
            if wig_src_points.shape[0] > max(idx_affine_wig) and dst_points.shape[0] > max(idx_affine_wig):
                srcW = wig_src_points[idx_affine_wig].astype(np.float32)
                dstW = dst_points[idx_affine_wig].astype(np.float32)
                M_wig, _ = cv2.estimateAffinePartial2D(srcW, dstW, method=cv2.LMEDS)
                if M_wig is not None:
                    wig_warp = warp_rgba_affine(wig, M_wig, W, H)
                    a = wig_warp[:, :, 3].astype(np.float32) / 255.0
                    a = a * hm
                    wig_warp[:, :, 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
                    frame = alpha_blend(frame, wig_warp, 0, 0)

            if args.debug:
                for p in dst_points[::25]:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)

            cv2.imshow("Masque + wig (q/ESC)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        hair_async.close()


if __name__ == "__main__":
    main()
