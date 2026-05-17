# =============================================================================
# RANDOM FOREST MONOCULAR DEPTH ESTIMATION
# =============================================================================

import cv2
import joblib
import argparse
import warnings
import numpy as np

from pathlib import Path
from scipy.stats import spearmanr
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity as ssim_fn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


IMPLEMENTATION_MODE = "custom"   

# =============================================================================
# PATHS & SETTINGS
# =============================================================================

NYU_IMG_DIR     = Path("rawImages")
GT_DIR          = Path("groundTruths")
OUTPUT_DIR      = Path("outputs")
MODEL_PATH      = Path("rf_depth_model.pkl")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_TRAIN_IMAGES  = 20
PIXELS_PER_IMG  = 5000
SAVE_COLORMAP   = True
CREATE_PARALLAX = True
PARALLAX_STR    = 18
SHOW_PREVIEW    = True


# =============================================================================
# HELPERS
# =============================================================================

def normalize_u8(arr):
    arr      = arr.astype(np.float32)
    lo, hi   = arr.min(), arr.max()
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


# =============================================================================
# ── CUSTOM IMPLEMENTATIONS ───────────────────────────────────────────────────
# Pure-NumPy versions of every DIP operator we use.
# These are mathematically identical to the OpenCV versions.
# =============================================================================

def _conv2d_manual(img, kernel):
    """Sliding-window 2-D convolution with reflect padding."""
    kh, kw       = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded        = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out           = np.zeros_like(img, dtype=np.float32)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y, x] = np.sum(padded[y:y+kh, x:x+kw] * kernel)
    return out


def to_gray_manual(bgr):
    """BGR → grayscale using ITU-R BT.601 luminance weights."""
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    return 0.114*b + 0.587*g + 0.299*r


def sobel_manual(gray_f):
    """Sobel edge detection — returns (gx, gy) gradient maps."""
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1,-2,-1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]], dtype=np.float32)
    return _conv2d_manual(gray_f, kx), _conv2d_manual(gray_f, ky)


def laplacian_manual(gray_f):
    """Laplacian operator — highlights regions of rapid intensity change."""
    k = np.array([[0,  1, 0],
                  [1, -4, 1],
                  [0,  1, 0]], dtype=np.float32)
    return _conv2d_manual(gray_f, k)


def gaussian_blur_manual(gray_f, ksize=7, sigma=0.0):
    """Gaussian blur — builds separable kernel then convolves."""
    if sigma <= 0.0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    ax       = np.arange(-ksize//2 + 1.0, ksize//2 + 1.0)
    k1d      = np.exp(-0.5 * (ax / sigma) ** 2)
    k1d     /= k1d.sum()
    k2d      = np.outer(k1d, k1d).astype(np.float32)
    return _conv2d_manual(gray_f, k2d)


def local_variance_manual(gray_f, size=7):
    """Local variance via E[X²] - E[X]² using uniform box filter."""
    # box filter = convolution with all-ones kernel / (size*size)
    box  = np.ones((size, size), dtype=np.float32) / (size * size)
    mean    = _conv2d_manual(gray_f, box)
    mean_sq = _conv2d_manual(gray_f ** 2, box)
    return mean_sq - mean ** 2


# =============================================================================
# ── BUILTIN IMPLEMENTATIONS (OpenCV) ─────────────────────────────────────────
# =============================================================================

def to_gray_builtin(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

def sobel_builtin(gray_f):
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy

def laplacian_builtin(gray_f):
    return cv2.Laplacian(gray_f, cv2.CV_32F)

def gaussian_blur_builtin(gray_f, ksize=7, sigma=0.0):
    return cv2.GaussianBlur(gray_f, (ksize, ksize), sigma)

def local_variance_builtin(gray_f, size=7):
    mean    = uniform_filter(gray_f,      size=size)
    mean_sq = uniform_filter(gray_f ** 2, size=size)
    return mean_sq - mean ** 2


# =============================================================================
# ── DISPATCHER — picks custom or builtin based on IMPLEMENTATION_MODE ─────────
# =============================================================================

def _to_gray(bgr):
    if IMPLEMENTATION_MODE == "custom":
        gray_f = to_gray_manual(bgr) / 255.0   # returns float, normalise
        return gray_f
    else:
        return to_gray_builtin(bgr) / 255.0

def _sobel(gray_f):
    if IMPLEMENTATION_MODE == "custom":
        return sobel_manual(gray_f)
    else:
        return sobel_builtin(gray_f)

def _laplacian(gray_f):
    if IMPLEMENTATION_MODE == "custom":
        return laplacian_manual(gray_f)
    else:
        return laplacian_builtin(gray_f)

def _local_variance(gray_f):
    if IMPLEMENTATION_MODE == "custom":
        return local_variance_manual(gray_f)
    else:
        return local_variance_builtin(gray_f)


# =============================================================================
# FEATURE EXTRACTION  — always calls _sobel / _laplacian / _local_variance
# so the mode switch propagates automatically
# =============================================================================

def extract_features(bgr):
    """
    Extracts 6 per-pixel DIP features.
    Which implementation runs depends on IMPLEMENTATION_MODE.

      0  intensity   — normalised grayscale
      1  grad_mag    — Sobel gradient magnitude  (edge / depth discontinuity)
      2  |laplacian| — second-order edges        (texture + focus cue)
      3  variance    — local intensity variance  (texture density)
      4  x-coord     — normalised x position     (spatial prior)
      5  y-coord     — normalised y position     (vertical perspective prior)
    """
    print(f"  [extract_features] using {IMPLEMENTATION_MODE} ops")

    gray_f   = _to_gray(bgr)                       # grayscale float [0,1]
    gx, gy   = _sobel(gray_f)                      # edge gradients
    grad_mag = np.sqrt(gx**2 + gy**2)
    lap      = _laplacian(gray_f)                  # second-order edges
    variance = _local_variance(gray_f)             # texture complexity

    h, w     = gray_f.shape
    yy, xx   = np.mgrid[0:h, 0:w]
    xx       = xx.astype(np.float32) / w
    yy       = yy.astype(np.float32) / h

    return np.stack([
        gray_f,
        grad_mag,
        np.abs(lap),
        variance,
        xx,
        yy
    ], axis=-1)


# =============================================================================
# IMAGE LOADING
# =============================================================================

def load_npy_as_bgr(path):
    arr = np.load(str(path))
    arr = normalize_u8(arr)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr


def load_gt(stem):
    number = stem.split("_")[-1]
    p      = GT_DIR / f"depth_{number}.npy"
    if not p.exists():
        return None
    gt     = np.load(str(p)).astype(np.float32)
    lo, hi = gt.min(), gt.max()
    return (gt - lo) / (hi - lo + 1e-9)


def smart_sample(X_list, y_list, n=8000):
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    idx = np.random.choice(len(X), min(n, len(X)), replace=False)
    return X[idx], y[idx]


# =============================================================================
# BUILD TRAINING DATA
# =============================================================================

def build_training_data(n_images, pixels_per_img):
    img_paths = sorted(NYU_IMG_DIR.glob("*.npy"))
    if not img_paths:
        raise FileNotFoundError(f"No .npy files in {NYU_IMG_DIR.resolve()}")

    X_all, y_all = [], []
    matched      = 0

    for img_path in img_paths[:n_images]:
        stem   = img_path.stem
        number = stem.split("_")[-1]
        gt_p   = GT_DIR / f"depth_{number}.npy"

        if not gt_p.exists():
            print(f"  [SKIP] no GT: {gt_p.name}")
            continue

        print(f"  [Train] {img_path.name}")
        matched += 1

        bgr  = load_npy_as_bgr(img_path)
        gt   = np.load(str(gt_p)).astype(np.float32)
        if gt.max() - gt.min() < 1e-8:
            continue
        gt   = (gt - gt.min()) / (gt.max() - gt.min())

        feat = extract_features(bgr)
        h, w, c = feat.shape
        X    = feat.reshape(-1, c)
        y    = gt.reshape(-1)
        idx  = np.random.choice(len(X), min(pixels_per_img, len(X)), replace=False)
        X_all.append(X[idx])
        y_all.append(y[idx])

    if not X_all:
        raise RuntimeError("No valid image/GT pairs found.")

    X_out, y_out = smart_sample(X_all, y_all)
    print(f"\n  Matched pairs  : {matched}")
    print(f"  Training pixels: {len(X_out)}")
    return X_out, y_out


# =============================================================================
# TRAIN MODEL
# =============================================================================

def train_model(X, y):
    print(f"\n[Training — mode: {IMPLEMENTATION_MODE}]\n")

    X = np.asarray(X).reshape(-1, np.asarray(X).shape[-1])
    y = np.asarray(y).reshape(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=100, max_depth=20,
        min_samples_leaf=4, max_features="sqrt",
        n_jobs=-1, random_state=42, verbose=1)

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"\n  Validation R²: {score:.4f}")

    names = ["intensity","gradient","laplacian","variance","x-coord","y-coord"]
    print("  Feature importances:")
    for n, i in sorted(zip(names, model.feature_importances_), key=lambda x: -x[1]):
        print(f"    {n:<14} {i:.4f}  {'█'*int(i*40)}")

    joblib.dump(model, MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")
    return model


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found — run training first.")
    print(f"Loading model: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


# =============================================================================
# DEPTH PREDICTION
# =============================================================================

def predict_depth(bgr, model):
    feat  = extract_features(bgr)
    h, w, c = feat.shape
    X     = feat.reshape(-1, c)
    pred  = model.predict(X)
    depth = np.clip(pred.reshape(h, w), 0.0, 1.0).astype(np.float32)
    return depth, normalize_u8(depth)


# =============================================================================
# PARALLAX
# =============================================================================

def create_parallax(bgr, depth_u8, strength):
    h, w  = depth_u8.shape
    shift = depth_u8.astype(np.float32) / 255.0 * strength
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                          np.arange(h, dtype=np.float32))
    map_x = np.clip(gx - shift, 0, w-1).astype(np.float32)
    return cv2.remap(bgr, map_x, gy.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(pred, gt):
    pred_f, gt_f = pred.astype(np.float32), gt.astype(np.float32)
    if pred_f.shape != gt_f.shape:
        pred_f = cv2.resize(pred_f, (gt_f.shape[1], gt_f.shape[0]))
    ssim_val = ssim_fn(pred_f, gt_f, data_range=1.0)
    mse      = np.mean((pred_f - gt_f)**2)
    return {
        "SSIM":     ssim_val,
        "PSNR":     10 * np.log10(1.0 / (mse + 1e-9)),
        "MAE":      np.mean(np.abs(pred_f - gt_f)),
        "RMSE":     float(np.sqrt(mse)),
        "Spearman": float(spearmanr(pred_f.ravel(), gt_f.ravel())[0])
    }


def print_metrics(stem, m):
    print(f"\n{'─'*60}")
    print(f"  METRICS — {stem}  [{IMPLEMENTATION_MODE} mode]")
    print(f"{'─'*60}")
    print(f"  SSIM       : {m['SSIM']:.4f}")
    print(f"  PSNR       : {m['PSNR']:.2f} dB")
    print(f"  MAE        : {m['MAE']:.4f}")
    print(f"  RMSE       : {m['RMSE']:.4f}")
    print(f"  Spearman ρ : {m['Spearman']:.4f}")
    print(f"{'─'*60}\n")


def print_output_summary(stem, depth_u8, depth_metres=None, parallax=None):
    print(f"\n{'─'*65}")
    print(f"  OUTPUT ANALYSIS — {stem}")
    print(f"{'─'*65}")
    total = depth_u8.size
    print(f"  Mean Depth Intensity : {np.mean(depth_u8):.2f}")
    print(f"  Std Depth            : {np.std(depth_u8):.2f}")
    print(f"  Near Pixels (>180)   : {np.sum(depth_u8>180)/total*100:.2f}%")
    print(f"  Far Pixels  (<70)    : {np.sum(depth_u8<70)/total*100:.2f}%")
    edges = cv2.Canny(depth_u8, 100, 200)
    print(f"  Edge Density         : {np.mean(edges>0)*100:.2f}%")
    if parallax is not None:
        print(f"  Parallax Generated   : YES")
    if depth_metres is not None:
        print(f"  Mean Depth (float)   : {np.mean(depth_metres):.4f}")
        print(f"  Max  Depth (float)   : {np.max(depth_metres):.4f}")
    print(f"{'─'*65}\n")


# =============================================================================
# RUN TRAIN / PREDICT
# =============================================================================

def run_train():
    X, y = build_training_data(N_TRAIN_IMAGES, PIXELS_PER_IMG)
    train_model(X, y)


def run_predict(model=None):
    if model is None:
        model = load_model()

    all_paths  = sorted(NYU_IMG_DIR.glob("*.npy"))
    n          = len(all_paths)
    test_paths = all_paths[int(n*0.8):] or all_paths

    if not test_paths:
        raise FileNotFoundError(f"No .npy images in {NYU_IMG_DIR.resolve()}")

    all_metrics = []

    for img_path in test_paths:
        print(f"\n[Predicting] {img_path.name}")
        bgr = load_npy_as_bgr(img_path)

        depth_metres, depth_u8 = predict_depth(bgr, model)
        stem = img_path.stem

        np.save(str(OUTPUT_DIR / f"{stem}.npy"), depth_metres)
        cv2.imwrite(str(OUTPUT_DIR / f"{stem}_vis.jpg"), depth_u8)
        print(f"  Saved vis    : outputs/{stem}_vis.jpg")

        if SAVE_COLORMAP:
            cmap = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(OUTPUT_DIR / f"{stem}_color.jpg"), cmap)

        par = None
        if CREATE_PARALLAX:
            par = create_parallax(bgr, depth_u8, PARALLAX_STR)
            cv2.imwrite(str(OUTPUT_DIR / f"{stem}_parallax.jpg"), par)

        print_output_summary(stem, depth_u8, depth_metres, par)

        gt = load_gt(stem)
        if gt is not None:
            m = compute_metrics(depth_metres, gt)
            all_metrics.append((stem, m))
            print_metrics(stem, m)
        else:
            print("  No GT found.")

        if SHOW_PREVIEW:
            cv2.imshow("Original", bgr)
            cv2.imshow("Depth", cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO))
            if par is not None:
                cv2.imshow("Parallax", par)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    if all_metrics:
        print(f"\n{'═'*75}")
        print(f"  FINAL SUMMARY  [{IMPLEMENTATION_MODE} mode]")
        print(f"{'═'*75}")
        for s, m in all_metrics:
            print(f"  {s:<15}  SSIM={m['SSIM']:.4f}  PSNR={m['PSNR']:.2f}  "
                  f"MAE={m['MAE']:.4f}  Spearman={m['Spearman']:.4f}")
        print(f"{'═'*75}\n")
    else:
        print(f"\nOutputs saved to {OUTPUT_DIR}/\n")


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF Depth Estimator")
    parser.add_argument("--mode", choices=["train","predict","both"], default="both")
    parser.add_argument("--impl", choices=["custom","builtin"], default=None,
                        help="Override IMPLEMENTATION_MODE from command line")
    args = parser.parse_args()

    # Command-line override
    if args.impl:
        IMPLEMENTATION_MODE = args.impl

    print(f"\n{'='*60}")
    print(f"  Implementation : {IMPLEMENTATION_MODE.upper()}")
    print(f"{'='*60}\n")

    if args.mode == "train":
        run_train()
    elif args.mode == "predict":
        run_predict()
    else:
        X, y  = build_training_data(N_TRAIN_IMAGES, PIXELS_PER_IMG)
        model = train_model(X, y)
        run_predict(model)