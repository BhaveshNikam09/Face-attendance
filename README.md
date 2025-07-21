# 🎯 Face Detection & Recognition using MTCNN + InsightFace

This project uses **MTCNN** for face detection and **InsightFace (ArcFace)** for high-accuracy face recognition. It supports real-time recognition from webcam/video and matching against a known face database.

---

## 🔧 Features

- Real-time face detection (MTCNN)
- Face recognition via embeddings (ArcFace)
- Match faces against a known database
- Works with webcam, image, or video input
- Easily add new people with a single image

---

## 🛠️ Technologies

- Python 3.8+
- MTCNN
- InsightFace
- OpenCV
- NumPy
- scikit-learn

---

---

## 🧑‍💻 Installation

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/BhaveshNikam09/Face-attendance.git
cd Face-attendance

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


python src/app.py

How It Works
1.MTCNN detects face & landmarks.

2.InsightFace (ArcFace) generates a 512-dim embedding.

3.Cosine similarity compares embeddings.

4.Closest match (below threshold) is returned.



---

This version will look **well-aligned and readable on GitHub**, mobile-friendly, and avoids markdown box overflows.

Want me to create a version with emoji badges or project screenshots too?
