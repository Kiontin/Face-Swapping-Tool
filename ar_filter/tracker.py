
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceLandmarkTracker:
    def __init__(
        self,
        model_path: str,
        num_faces: int = 1,
        min_det: float = 0.5,
        min_presence: float = 0.5,
        min_track: float = 0.5,
        use_gpu: bool = False,
    ):
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

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=min_det,
            min_face_presence_confidence=min_presence,
            min_tracking_confidence=min_track,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self):
        self._landmarker.close()

    def get_landmarks_points(self, frame_bgr: np.ndarray):
        H, W = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)

        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        if not result.face_landmarks:
            return None

        lms = result.face_landmarks[0]
        pts = np.array([(lm.x * W, lm.y * H) for lm in lms], dtype=np.float32)
        return pts
