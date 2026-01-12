#!/usr/bin/env bash
set -e
mkdir -p assets

echo "Téléchargement face_landmarker.task..."
curl -L -o assets/face_landmarker.task \
"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

echo "Téléchargement selfie_multiclass_256x256.tflite..."
curl -L -o assets/selfie_multiclass_256x256.tflite \
"https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"

echo "OK ✅"
