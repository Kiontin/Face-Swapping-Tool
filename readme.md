# AR Stylized Mask + Hair (MediaPipe)

Filtre AR **non-identitaire** : applique un **masque stylisé** sur le visage et une **perruque** (cheveux extraits du masque PNG), en temps réel via webcam.

> Chaque utilisateur ajoute son propre `assets/mask.png` (PNG RGBA).

---

## Démo
- Face tracking : **MediaPipe Face Landmarker**
- Hair segmentation : **MediaPipe Image Segmenter (selfie_multiclass)**

---

## Prérequis
- Python 3.9+ recommandé
- macOS / Windows / Linux
- Git installé (VS Code utilise le Git de la machine)  
  Voir la doc VS Code : Git requis pour activer le Source Control.  
  https://code.visualstudio.com/docs/sourcecontrol/overview [1](https://chuoling.github.io/mediapipe/solutions/pose.html)

---

## Installation (local)
### 1) Créer et activer un venv
```bash
python3 -m venv .venv
source .venv/bin/activate
