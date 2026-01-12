
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

AVATAR_PNG = "assets/mask.png"
FACE_MODEL = "assets/face_landmarker.task"
SEG_MODEL  = "assets/selfie_multiclass_256x256.tflite"

OUT_WIG = "assets/wig.png"
OUT_WIG_POINTS = "assets/wig_src_points.npy"

HAIR_CLASS = 1  # multiclass: [background, hair, body, face, clothes, others] [3](https://github.com/spmallick/learnopencv/blob/master/SeamlessCloning/clone.py)[2](https://colab.research.google.com/github/bigvisionai/opencv-webinar-poisson-image-editing/blob/main/Seamless_Cloning.ipynb)


def load_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        a = np.full((img.shape[0], img.shape[1], 1), 255, np.uint8)
        img = np.concatenate([img, a], axis=2)
    if img.shape[2] != 4:
        raise ValueError("PNG doit être BGRA/RGBA (4 canaux).")
    return img


def rgba_to_rgb_on_white(rgba_bgra):
    bgr = rgba_bgra[:, :, :3].astype(np.float32)
    a = rgba_bgra[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full_like(bgr, 255.0)
    comp = bgr * a + white * (1 - a)
    rgb = cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return rgb


def main():
    avatar = load_rgba(AVATAR_PNG)
    H, W = avatar.shape[:2]
    rgb = rgba_to_rgb_on_white(avatar)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Segmentation multiclass -> hair mask (IMAGE) [2](https://colab.research.google.com/github/bigvisionai/opencv-webinar-poisson-image-editing/blob/main/Seamless_Cloning.ipynb)[3](https://github.com/spmallick/learnopencv/blob/master/SeamlessCloning/clone.py)
    base_seg = python.BaseOptions(model_asset_path=SEG_MODEL)
    seg_opts = vision.ImageSegmenterOptions(
        base_options=base_seg,
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True
    )
    segmenter = vision.ImageSegmenter.create_from_options(seg_opts)
    seg_res = segmenter.segment(mp_image)
    segmenter.close()

    cm = seg_res.category_mask.numpy_view().astype(np.uint8)
    hair = (cm == HAIR_CLASS).astype(np.float32)
    hair = cv2.GaussianBlur(hair, (31, 31), 0)
    hair = np.clip(hair, 0.0, 1.0)

    wig = avatar.copy()
    a = wig[:, :, 3].astype(np.float32) / 255.0
    wig[:, :, 3] = (np.clip(a * hair, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(OUT_WIG, wig)

    # Landmarks du PNG (IMAGE) pour aligner la perruque [1](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker)
    base_face = python.BaseOptions(model_asset_path=FACE_MODEL)
    face_opts = vision.FaceLandmarkerOptions(
        base_options=base_face,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(face_opts)
    res = landmarker.detect(mp_image)
    landmarker.close()

    if not res.face_landmarks:
        raise RuntimeError("Aucun visage détecté sur mask.png")

    lms = res.face_landmarks[0]
    pts = np.array([(lm.x * W, lm.y * H) for lm in lms], dtype=np.float32)
    np.save(OUT_WIG_POINTS, pts)

    print("✅ OK")
    print(" - wig:", OUT_WIG)
    print(" - wig_src_points:", OUT_WIG_POINTS)


if __name__ == "__main__":
    main()
