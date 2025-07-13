# Reidentification in a Single feed
Player Re-Identification & Tracking with YOLOv8 and ResNet50 Embeddings

📋 Overview

This project demonstrates a robust pipeline for detecting and re-identifying multiple players in video streams using:

YOLOv8 (Ultralytics) for high-precision player detection.

ResNet50 backbone to extract 128-dimensional embeddings for appearance-based re-identification (ReID).

A simple yet effective greedy tracker that assigns consistent IDs across frames using cosine similarity.

Together, these components enable accurate multi-object tracking (MOT) tailored to sports and surveillance applications.

🚀 Features

Real-time detection & tracking (CPU & GPU support)

Appearance-based re-identification to handle occlusions and re-entries

Configurable similarity threshold for sensitivity control

Template update strategy: smooth adaptation by mixing historical and current embeddings

Extensible codebase: easily swap detection model, backbone, or tracker logic

⚙️ Requirements

Python 3.8+

PyTorch 1.13+

OpenCV 4.5+

Ultralytics YOLOv8 (ultralytics package)

torchvision

numpy

To install dependencies, run:

pip install torch torchvision opencv-python numpy ultralytics

GPU Users: Ensure CUDA drivers and torch installation are compatible (e.g., pip install torch --extra-index-url https://download.pytorch.org/whl/cu121).


💡 How It Works

Detection

Load a fine-tuned YOLOv8 model (best.pt) for player detection.

Process each video frame, obtaining bounding boxes, confidence scores, and class IDs.

Embedding Extraction

Crop detected player regions and resize to 224×224.

Pass through a ResNet50 backbone (without final classification layer) + a linear head to obtain normalized 128-D vectors.

Re-Identification & Tracking

Maintain a dictionary of templates: {track_id: embedding}.

For each frame, compute cosine similarities between new embeddings and templates.

Greedily match pairs above a similarity threshold.

Update matched templates by averaging old & new embeddings (50/50 mix).

Assign new IDs to unmatched embeddings.

Visualization

Draw bounding boxes & track IDs (P1, P2, ...) on frames.

Save processed frames into tracked_output.mp4.

📝 Usage

Clone the repository:

git clone https://github.com/your-username/player-reid-tracker.git
cd player-reid-tracker



python src/detect.py --model models/best.pt --input data/input.mp4 --output data/tracked_output.mp4 --device cuda --threshold 0.6

**Inspect results**:
   - View `data/tracked_output.mp4` to see tracked players with consistent IDs.

---

## 📈 Performance & Tuning

- **Threshold selection**: Experiment with similarity thresholds in the range `[0.4, 0.8]`. Lower values yield more ID switches; higher values may drop true matches.
- **Speed**: CPU processing ~1 FPS; using GPU can increase to ~10–15 FPS depending on hardware.
- **Batch size**: Increase batch inference of embedding network for faster GPU throughput.

---

## 📂 Extensibility

- Swap out `EmbeddingNet` with other ReID backbones (e.g., ResNet101, MobileNet).
- Integrate advanced trackers (e.g., SORT, DeepSORT) by replacing `ReIDTracker`.
- Extend to multi-class scenarios by filtering multiple YOLO classes.

---
