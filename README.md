# Mars Terrain Hazard Detection (YOLOv8)

*Robot vision fundamentals project, Aug 2025 (Excellence Award). Uploaded to GitHub in August 2026.*

YOLOv8n vs. YOLOv8s on AI4Mars-style rover imagery. Does the lighter model actually lose anything on a small (322-image) dataset? Or is the bigger model's extra capacity wasted here too?

## Motivation

A rover navigating Mars terrain needs to spot hazards (large rocks) and terrain types (sand, soil, bedrock) fast, and on limited onboard hardware. This compares a nano and a small YOLOv8 detector on the same task, to see whether the smaller, faster, onboard-friendlier model actually gives up meaningful accuracy. It continues the same "does bigger actually help" question as the other model-comparison projects in this portfolio.

## Approach

- **Data:** AI4Mars-style rover imagery (Roboflow-hosted export), with region labels converted into YOLO detection boxes. 322 usable images, split 70/20/10 (train 225, valid 64, test 33).
- **Classes:** bedrock, big-rock, sand, soil. A 5th candidate class (`terrain`) was defined too, but had zero annotated instances in this dataset, so it was dropped automatically.
- **Models:** YOLOv8n (3.0M params) vs. YOLOv8s (11.1M params). Same data, same seed, 50-epoch budget with early stopping (patience 10).

## Results

*(measured on the validation split, see [Limitations](#limitations--whats-next) for why not the held-out test split)*

| | YOLOv8n | YOLOv8s |
|---|---|---|
| mAP50 | **0.485** | 0.476 |
| mAP50-95 | **0.376** | 0.369 |
| Precision | 0.504 | 0.562 |
| Recall | **0.512** | 0.440 |
| Model size | **6.3MB** | 22.5MB |
| Inference speed | **4.7ms** | 11.7ms |

**Per-class mAP50 (YOLOv8n):**

| Class | Images | Instances | mAP50 |
|---|---|---|---|
| soil | 33 | 36 | 0.849 |
| sand | 22 | 30 | 0.551 |
| bedrock | 30 | 90 | 0.477 |
| big-rock | 8 | 41 | **0.060** |

**Takeaway:** on this dataset, the smaller model wins on mAP50, mAP50-95, and recall, while being 3.6x smaller and 2.5x faster. YOLOv8s only wins on precision. The bigger story is really per-class: `big-rock` (the actual hazard class this project cares most about) has only 8 training images, and its detector basically failed (mAP50 0.060). Meanwhile `soil`, the best-represented class, reached 0.849. At that per-class level, this isn't really about model capacity. It's about data scarcity. More big-rock training images would likely matter more than a bigger model would.

## Tech stack

Python, Ultralytics YOLOv8, Roboflow (dataset hosting/export), scikit-learn (stratified split), PyYAML, OpenCV, Matplotlib.

## How to run

```bash
pip install roboflow ultralytics scikit-learn pyyaml
```

Needs a Roboflow API key (set the `ROBOFLOW_API_KEY` env var, or a Colab secret named `ROBOFLOW`) with access to the `s-workspace-cfuov/ai4mars-sbhtm` project. The dataset itself isn't bundled in this repo. Open `mars_terrain_detection.ipynb` and run it top to bottom (originally built for Google Colab; GPU recommended for training).

## Limitations / what's next

- **The results above are validation-split numbers, not held-out test-split numbers.** A 33-image test split was created (70/20/10) but never evaluated in the original training run. Only train/valid were used, so the final report is really valid-split performance. This notebook (the version committed here) fixes that gap structurally: it now evaluates on val *and* the held-out test split, and prints metrics straight from the result objects instead of hard-coding them. But it hasn't been re-run since that fix, so the numbers above are still the original valid-split results, not yet confirmed on genuinely unseen test data.
- `big-rock` detection isn't usable as it stands (mAP50 0.060). 8 training images just isn't enough signal. More labeled big-rock examples, or a class-weighted loss, would likely help more than any architecture change would.
- Converting region masks into bounding boxes is a real task change, not a lossless export (this is noted directly in the code). It under-represents area hazards and over-penalizes large, irregularly-shaped polygons. A segmentation model (`yolov8n-seg`) would be a more faithful match to the original AI4Mars-style region labels, if terrain *coverage* (not just discrete hazard boxes) is actually the goal.

## Credits

Built independently. Dataset re-hosted via Roboflow in an AI4Mars-style COCO-segmentation export. Original AI4Mars dataset from NASA JPL.
