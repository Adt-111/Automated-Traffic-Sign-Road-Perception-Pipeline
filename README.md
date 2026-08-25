# Automated Traffic Sign & Road Perception Pipeline

A two-stage traffic sign detection + classification pipeline:

1. **Classical detector** (`src/classical_detector.py`) — HSV color masking
   (red/blue/yellow) reinforced with Sobel edges, contour extraction, and a
   SIFT + Bag-of-Visual-Words + SVM baseline classifier.
2. **CNN classifier** (`src/cnn_classifier.py`) — a custom 3-block
   Conv/BatchNorm/ReLU/MaxPool/Dropout CNN trained on GTSRB (or a synthetic
   fallback dataset), with loss-curve and confusion-matrix reporting.
3. **Streamlit dashboard** (`app.py`) — upload an image or video; the
   classical detector proposes candidate sign regions, the CNN classifies
   each crop, and results are overlaid as labeled boxes with confidence.

## Project layout

```
traffic_sign_perception/
├── requirements.txt
├── config.py                  # paths, GTSRB class names, all hyperparameters
├── train_cnn.py                # standalone CNN training entry point
├── app.py                      # Streamlit dashboard
└── src/
    ├── classical_detector.py   # HSV masking, Sobel/contours, SIFT+SVM baseline
    └── cnn_classifier.py       # TrafficSignCNN architecture, training, eval, plots
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Train the CNN. Auto-detects GTSRB at $GTSRB_ROOT/Train; falls back to a
#    fast synthetic 5-class demo dataset if no real data is found.
python train_cnn.py --synthetic --epochs 5   # quick demo run, no dataset download needed

# 2. Launch the dashboard
streamlit run app.py
```

## Using the real GTSRB dataset

1. Download the Kaggle "GTSRB - German Traffic Sign Recognition Benchmark"
   dataset (folder layout: `Train/<class_id>/*.png`, `Test/*.png` + `Test.csv`).
2. Point the project at it:
   ```bash
   export GTSRB_ROOT=/path/to/gtsrb
   python train_cnn.py --epochs 15
   ```
3. `train_cnn.py` will automatically use all 43 GTSRB classes (`config.NUM_CLASSES`)
   instead of the synthetic 5-class demo set, save the best checkpoint to
   `checkpoints/cnn_classifier.pth`, and write `training_curves.png` +
   `confusion_matrix.png` to `checkpoints/`.
4. `streamlit run app.py` will pick up that checkpoint automatically.

> **Note:** if you train on the synthetic demo dataset (5 classes) and then
> run `app.py` with the default `config.NUM_CLASSES = 43`, you'll get a
> state-dict size mismatch — that's expected. Either train on real GTSRB
> data before using the app for real classification, or temporarily set
> `config.NUM_CLASSES = 5` to demo the app end-to-end with the synthetic
> checkpoint.

## Module-level design notes

- **HSV masking** (`build_color_mask`): red wraps around hue 0 in OpenCV's
  HSV space, so it's handled as two separate ranges that get OR-ed together.
  Morphological close/open cleans up speckle noise and fills small gaps
  inside sign regions (e.g. a pictogram interrupting a solid color fill).
- **Edge reinforcement**: `find_candidate_regions` ANDs the color mask with
  a dilated Sobel edge map before contour-finding, so a hazy/color-similar
  background region without a real boundary won't produce a candidate box.
  The "how solid is this box" `mask_score` check, however, is computed
  against the *original* color mask — ANDing with the edge ring would make
  every solid-color sign score artificially low.
- **Non-max suppression**: candidates from different color channels that
  overlap significantly (IoU > 0.3) are deduplicated, keeping the one with
  the highest `mask_score`.
- **SIFT + BoVW + SVM** (`SIFTBagOfWordsSVM`): a from-scratch bag-of-visual-
  words pipeline — SIFT descriptors are clustered into a visual vocabulary
  via MiniBatchKMeans, each image becomes a normalized histogram of visual-
  word occurrences, and a linear-kernel-free RBF SVM classifies the
  histogram. This is a genuinely non-deep baseline, useful for a report
  section contrasting classical vs. learned features.
- **TrafficSignCNN**: 3x (Conv → BatchNorm → ReLU → MaxPool → Dropout)
  blocks, with the flattened feature dimension inferred dynamically from
  `config.CNN_IMAGE_SIZE` so changing the input resolution doesn't require
  manually recomputing a `Linear` layer's input size.
- **Training**: `ReduceLROnPlateau` on validation loss, best-checkpoint
  saving by validation accuracy, and light data augmentation (color jitter
  + random affine) applied only to the training split.
- **app.py**: `@st.cache_resource` ensures the CNN is loaded once per
  session, not on every interaction. The pipeline degrades gracefully to
  "detection only" mode (placeholder labels) if no trained checkpoint
  exists yet, rather than crashing.

## Testing individual stages

```bash
python -m src.classical_detector   # synthetic road scene -> candidate boxes
python -m src.cnn_classifier       # synthetic 5-class dataset -> trains 3 epochs, plots curves
```

Both scripts print progress and save example outputs to `/tmp/` for a quick
sanity check without needing GTSRB or a GPU.
