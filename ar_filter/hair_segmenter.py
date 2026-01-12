
import time
import threading
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Modèle multiclass: [background, hair, body, face, clothes, others]
HAIR_CLASS_INDEX = 1


def _smoothing_factor(te, cutoff):
    r = 2.0 * np.pi * cutoff * te
    return r / (r + 1.0)


def _exponential_smoothing(a, x, x_prev):
    return a * x + (1.0 - a) * x_prev


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.03, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def reset(self):
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


class HairSegmenter:
    """
    ImageSegmenter (VIDEO) -> confidence mask cheveux float32.
    Les confidence masks sont une sortie prévue par l'Image Segmenter. [1](https://colab.research.google.com/github/bigvisionai/opencv-webinar-poisson-image-editing/blob/main/Seamless_Cloning.ipynb)[2](https://stackoverflow.com/questions/77643447/access-specific-pose-landmarker-task-api-on-python)
    """
    def __init__(self, model_path: str, use_gpu: bool = True):
        if use_gpu:
            try:
                base_options = python.BaseOptions(
                    model_asset_path=model_path,
                    delegate=python.BaseOptions.Delegate.GPU
                )
            except Exception:
                base_options = python.BaseOptions(model_asset_path=model_path)
        else:
            base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)

    def close(self):
        self._segmenter.close()

    def hair_confidence01(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)
        res = self._segmenter.segment_for_video(mp_image, ts_ms)

        m = res.confidence_masks[HAIR_CLASS_INDEX].numpy_view()
        m = m.astype(np.float32)

        # ✅ garantie (H,W)
        if m.ndim == 3 and m.shape[2] == 1:
            m = m[:, :, 0]
        return m


class AsyncHairMask:
    def __init__(self, model_path: str, use_gpu: bool = True,
                 min_cutoff=1.0, beta=0.03, d_cutoff=1.0):
        self._seg = HairSegmenter(model_path, use_gpu=use_gpu)
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_mask = None
        self._running = True
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

        self._filt = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def submit(self, frame_bgr: np.ndarray):
        with self._lock:
            self._latest_frame = frame_bgr

    def get_mask(self):
        with self._lock:
            return None if self._latest_mask is None else self._latest_mask.copy()

    def _loop(self):
        while self._running:
            frame = None
            with self._lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame
                    self._latest_frame = None

            if frame is None:
                time.sleep(0.001)
                continue

            try:
                m = self._seg.hair_confidence01(frame)
                m = np.clip(m, 0.0, 1.0)
                m = self._filt(m, t=time.time())

                # ✅ garantie (H,W)
                if m.ndim == 3 and m.shape[2] == 1:
                    m = m[:, :, 0]

                with self._lock:
                    self._latest_mask = m
            except Exception:
                time.sleep(0.005)

    def close(self):
        self._running = False
        try:
            self._worker.join(timeout=0.5)
        except Exception:
            pass
        self._seg.close()
