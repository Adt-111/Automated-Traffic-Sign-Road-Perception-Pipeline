"""
The non-deep-learning half of the pipeline: HSV masking to find red/blue/
yellow regions, Sobel edges + contours to turn those into candidate boxes,
and a SIFT + Bag-of-Visual-Words + SVM classifier as a baseline that
doesn't need a GPU (or the CNN in cnn_classifier.py) to run.

Everything here works on plain OpenCV BGR arrays, same as cv2.imread.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config


# 1. HSV color masking

def build_color_mask(image_bgr: np.ndarray, colors: list[str] | None = None) -> np.ndarray:
    """
    Binary mask marking pixels in the requested colors (default: red,
    blue, yellow -- covers most sign types). colors defaults to all three;
    image_bgr should be a standard cv2.imread BGR array. Returns an (H, W)
    uint8 mask, 255 where a pixel matched.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("build_color_mask received an empty or None image.")

    colors = colors or list(config.HSV_RANGES.keys())
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for color in colors:
        if color not in config.HSV_RANGES:
            raise ValueError(f"Unknown color '{color}'. Expected one of {list(config.HSV_RANGES)}.")
        for band in config.HSV_RANGES[color]:
            lower = np.array(band["lower"], dtype=np.uint8)
            upper = np.array(band["upper"], dtype=np.uint8)
            band_mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, band_mask)

    # Morphological close + open to remove speckle noise and fill small gaps
    # inside sign regions (e.g. text/pictogram interrupting a solid color mask).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return combined_mask


# 2. Sobel edge detection + contour-based bounding box extraction

def compute_sobel_edges(image_bgr: np.ndarray) -> np.ndarray:
    """
    Computes a normalized Sobel gradient-magnitude edge map from the
    grayscale image. Used to reinforce sign boundaries that the color mask
    alone might miss (e.g. faded paint, partial occlusion).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, dx=0, dy=1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return normalized


@dataclass
class Candidate:
    """A single candidate sign region: its bounding box and the color that triggered it."""
    x: int
    y: int
    w: int
    h: int
    color: str
    mask_score: float  # fraction of the bbox that was masked as sign-colored (0-1)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def crop(self, image_bgr: np.ndarray, padding_ratio: float = config.CROP_PADDING_RATIO) -> np.ndarray:
        """Returns the (padded) crop of `image_bgr` corresponding to this candidate's bbox."""
        H, W = image_bgr.shape[:2]
        pad_w, pad_h = int(self.w * padding_ratio), int(self.h * padding_ratio)
        x0 = max(0, self.x - pad_w)
        y0 = max(0, self.y - pad_h)
        x1 = min(W, self.x + self.w + pad_w)
        y1 = min(H, self.y + self.h + pad_h)
        return image_bgr[y0:y1, x0:x1]


def find_candidate_regions(
    image_bgr: np.ndarray,
    colors: list[str] | None = None,
    use_edge_reinforcement: bool = True,
) -> list[Candidate]:
    """
    Color-masks the image, reinforces it with Sobel edges (so a hazy
    reddish sky doesn't get flagged just for having the right hue), finds
    contours, and filters down to plausible sign-shaped boxes. Returns a
    deduplicated list of Candidates (NMS across colors).
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("find_candidate_regions received an empty or None image.")

    colors = colors or list(config.HSV_RANGES.keys())
    H, W = image_bgr.shape[:2]
    image_area = H * W

    edge_map = None
    if use_edge_reinforcement:
        edges = compute_sobel_edges(image_bgr)
        _, edge_binary = cv2.threshold(edges, 40, 255, cv2.THRESH_BINARY)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edge_map = cv2.dilate(edge_binary, dilate_kernel, iterations=1)

    all_candidates: list[Candidate] = []

    for color in colors:
        color_mask = build_color_mask(image_bgr, colors=[color])

        # Edge reinforcement is used only to VALIDATE that a candidate region has a real
        # boundary (filters out e.g. a hazy red-tinted sky), so contour-finding runs on the
        # edge-ANDed mask. Scoring how "solid" the region is, however, must use the original
        # color mask — ANDing with a (dilated) edge ring would otherwise turn every solid-color
        # blob into a thin boundary ring and make mask_score artificially low for any real sign.
        detection_mask = cv2.bitwise_and(color_mask, edge_map) if edge_map is not None else color_mask

        contours, _ = cv2.findContours(detection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < config.MIN_CONTOUR_AREA or area > config.MAX_CONTOUR_AREA_RATIO * image_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue
            aspect_ratio = w / h
            if not (config.MIN_ASPECT_RATIO <= aspect_ratio <= config.MAX_ASPECT_RATIO):
                continue

            bbox_color_region = color_mask[y : y + h, x : x + w]
            mask_score = float(np.count_nonzero(bbox_color_region)) / max(bbox_color_region.size, 1)
            if mask_score < 0.25:  # bounding box mostly not sign-colored -> likely a spurious contour
                continue

            all_candidates.append(Candidate(x=x, y=y, w=w, h=h, color=color, mask_score=mask_score))

    return _non_max_suppression(all_candidates, iou_threshold=0.3)


def _iou(a: Candidate, b: Candidate) -> float:
    ax0, ay0, ax1, ay1 = a.x, a.y, a.x + a.w, a.y + a.h
    bx0, by0, bx1, by1 = b.x, b.y, b.x + b.w, b.y + b.h

    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    inter_area = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
    if inter_area == 0:
        return 0.0

    union_area = a.w * a.h + b.w * b.h - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _non_max_suppression(candidates: list[Candidate], iou_threshold: float = 0.3) -> list[Candidate]:
    """Greedy NMS keyed on mask_score, so the most color-confident box wins each overlapping cluster."""
    ordered = sorted(candidates, key=lambda c: -c.mask_score)
    kept: list[Candidate] = []
    for cand in ordered:
        if all(_iou(cand, k) < iou_threshold for k in kept):
            kept.append(cand)
    return kept


def draw_candidates(image_bgr: np.ndarray, candidates: list[Candidate]) -> np.ndarray:
    """Returns a copy of `image_bgr` with candidate bounding boxes drawn (color-coded by detected hue)."""
    color_map = {"red": (0, 0, 255), "blue": (255, 0, 0), "yellow": (0, 255, 255)}
    output = image_bgr.copy()
    for cand in candidates:
        draw_color = color_map.get(cand.color, (0, 255, 0))
        cv2.rectangle(output, (cand.x, cand.y), (cand.x + cand.w, cand.y + cand.h), draw_color, 2)
    return output


# 3. SIFT feature extraction + Bag-of-Visual-Words + SVM baseline

class SIFTBagOfWordsSVM:
    """
    SIFT descriptors -> k-means visual vocabulary -> per-image histogram ->
    SVM. A CPU-only baseline the CNN should beat, no training required
    beyond fitting the vocabulary and the classifier itself.
    """

    def __init__(self, vocab_size: int = config.BOW_VOCAB_SIZE, n_features: int = config.SIFT_N_FEATURES) -> None:
        self.vocab_size = vocab_size
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.kmeans: MiniBatchKMeans | None = None
        self.scaler = StandardScaler()
        self.svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)
        self._is_fitted = False

    def _extract_descriptors(self, image_bgr: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, descriptors = self.sift.detectAndCompute(gray, None)
        return descriptors

    def _image_to_histogram(self, image_bgr: np.ndarray) -> np.ndarray:
        """Assigns each SIFT descriptor to its nearest visual word and returns the normalized histogram."""
        descriptors = self._extract_descriptors(image_bgr)
        histogram = np.zeros(self.vocab_size, dtype=np.float32)
        if descriptors is None or len(descriptors) == 0 or self.kmeans is None:
            return histogram
        words = self.kmeans.predict(descriptors)
        for w in words:
            histogram[w] += 1
        # L1-normalize so histograms are comparable across images with different keypoint counts.
        total = histogram.sum()
        if total > 0:
            histogram /= total
        return histogram

    def fit(self, images: list[np.ndarray], labels: list[int]) -> "SIFTBagOfWordsSVM":
        """
        Args:
            images: list of BGR cropped sign patches.
            labels: parallel list of integer class ids.
        """
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length.")

        all_descriptors = []
        for img in images:
            descriptors = self._extract_descriptors(img)
            if descriptors is not None and len(descriptors) > 0:
                all_descriptors.append(descriptors)

        if not all_descriptors:
            raise RuntimeError(
                "No SIFT descriptors could be extracted from any training image; "
                "check that images are non-trivial (not blank/uniform patches)."
            )

        stacked = np.vstack(all_descriptors)
        effective_k = min(self.vocab_size, len(stacked))
        self.kmeans = MiniBatchKMeans(n_clusters=effective_k, random_state=config.SEED, n_init=3)
        self.kmeans.fit(stacked)
        self.vocab_size = effective_k

        features = np.stack([self._image_to_histogram(img) for img in images])
        features = self.scaler.fit_transform(features)
        self.svm.fit(features, labels)
        self._is_fitted = True
        return self

    def predict(self, image_bgr: np.ndarray) -> tuple[int, float]:
        """Returns (predicted_class_id, confidence) for a single cropped sign patch."""
        if not self._is_fitted:
            raise RuntimeError("SIFTBagOfWordsSVM.predict called before fit(). Train or load a model first.")
        histogram = self._image_to_histogram(image_bgr).reshape(1, -1)
        histogram = self.scaler.transform(histogram)
        proba = self.svm.predict_proba(histogram)[0]
        pred_class = int(self.svm.classes_[np.argmax(proba)])
        confidence = float(np.max(proba))
        return pred_class, confidence

    def save(self, path: str = config.SVM_MODEL_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"vocab_size": self.vocab_size, "kmeans": self.kmeans, "scaler": self.scaler, "svm": self.svm},
                f,
            )

    @classmethod
    def load(cls, path: str = config.SVM_MODEL_PATH) -> "SIFTBagOfWordsSVM":
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(vocab_size=data["vocab_size"])
        model.kmeans = data["kmeans"]
        model.scaler = data["scaler"]
        model.svm = data["svm"]
        model._is_fitted = True
        return model


if __name__ == "__main__":  # pragma: no cover
    print("Running classical_detector.py smoke test with a synthetic road scene...")

    # Build a synthetic 400x300 "road scene" with a red octagon (stop-sign-like)
    # and a blue circle (mandatory-sign-like) painted on a gray background.
    scene = np.full((300, 400, 3), 120, dtype=np.uint8)  # gray "road/sky" background
    cv2.circle(scene, (100, 100), 40, (255, 0, 0), thickness=-1)   # blue circle (BGR: pure blue)
    cv2.rectangle(scene, (250, 180), (330, 260), (0, 0, 255), thickness=-1)  # red square-ish region
    scene = cv2.GaussianBlur(scene, (3, 3), 0)  # slight blur so edges aren't perfectly binary

    candidates = find_candidate_regions(scene)
    print(f"Found {len(candidates)} candidate region(s):")
    for c in candidates:
        print(f"  color={c.color:6s} bbox={c.bbox} mask_score={c.mask_score:.2f}")

    annotated = draw_candidates(scene, candidates)
    cv2.imwrite("/tmp/classical_detector_smoke_test.png", annotated)
    print("Saved annotated scene to /tmp/classical_detector_smoke_test.png")

    # SIFT+SVM smoke test on a few synthetic crops with distinguishable texture.
    print("\nTesting SIFTBagOfWordsSVM on synthetic textured patches...")
    rng = np.random.default_rng(0)
    synth_images, synth_labels = [], []
    for label in range(3):
        for _ in range(15):
            patch = np.zeros((64, 64, 3), dtype=np.uint8)
            # Give each class a distinct checkerboard frequency so SIFT keypoints differ.
            step = 8 + label * 6
            patch[::step, :] = 255
            patch[:, ::step] = 255
            noise = rng.integers(0, 40, size=patch.shape, dtype=np.uint8)
            patch = cv2.add(patch, noise)
            synth_images.append(patch)
            synth_labels.append(label)

    bow_svm = SIFTBagOfWordsSVM(vocab_size=20)
    bow_svm.fit(synth_images, synth_labels)
    pred_class, confidence = bow_svm.predict(synth_images[0])
    print(f"Prediction for a class-0 training patch: class={pred_class}, confidence={confidence:.2f}")
    print("classical_detector.py smoke test passed.")
