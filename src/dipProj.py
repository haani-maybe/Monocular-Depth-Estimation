# =============================================================================
# RANDOM FOREST MONOCULAR DEPTH ESTIMATION
# =============================================================================
#
# Features:
#   • Random Forest depth prediction
#   • Classical DIP feature extraction
#   • Depth map generation
#   • INFERNO colormap output
#   • Parallax pseudo-3D generation
#   • Ground-truth comparison
#   • Output statistics / summaries
#
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

import argparse

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

NYU_IMG_DIR = BASE_DIR / "rawImages"
GT_DIR = BASE_DIR / "groundTruths"
OUTPUT_DIR  = BASE_DIR / "outputs"
MODEL_PATH  = BASE_DIR / "models" / "rf_depth_model.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SETTINGS
# =============================================================================

N_TRAIN_IMAGES = 20
PIXELS_PER_IMG = 5000

SAVE_COLORMAP  = True
CREATE_PARALLAX = True
PARALLAX_STR    = 18

SHOW_PREVIEW = True

# =============================================================================
# HELPERS
# =============================================================================


def normalize_u8(arr):

    arr = arr.astype(np.float32)

    lo = arr.min()
    hi = arr.max()

    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)

    arr = (arr - lo) / (hi - lo)
    arr *= 255.0

    return arr.astype(np.uint8)


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
    # loads matching ground-truth depth map using filename index

    number = stem.split("_")[-1]
    p = GT_DIR / f"depth_{number}.npy"

    if not p.exists():
        return None

    gt = np.load(str(p)).astype(np.float32)

    # normalize gt depth to 0–1 range for stable metric computation
    lo = gt.min()
    hi = gt.max()

    gt = (gt - lo) / (hi - lo + 1e-9)

    return gt


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
def smart_sample(X, y, n=8000):
    # random pixel sampling to reduce training complexity

    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) == 0:
        return X, y

    n = min(n, len(X))
    idx = np.random.choice(len(X), n, replace=False)

    return X[idx], y[idx]

def extract_features(bgr):
    # core feature engineering step for classical depth estimation

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # normalize intensity to 0–1 for stable gradients
    gray_f = gray.astype(np.float32) / 255.0

    # sobel gradient -> captures edge strength (depth discontinuities)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

    grad_mag = np.sqrt(gx**2 + gy**2)

    # laplacian -> second-order edges (texture + depth transitions)
    lap = cv2.Laplacian(gray_f, cv2.CV_32F)

    # local variance -> texture complexity indicator (flat vs detailed regions)
    mean = uniform_filter(gray_f, size=7)
    mean_sq = uniform_filter(gray_f**2, size=7)
    variance = mean_sq - mean**2

    # normalized pixel coordinates -> gives spatial priors (top/bottom depth bias)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]

    xx = xx.astype(np.float32) / w
    yy = yy.astype(np.float32) / h

    # stack all per-pixel features into feature tensor
    features = np.stack([
        gray_f,
        grad_mag,
        np.abs(lap),
        variance,
        xx,
        yy
    ], axis=-1)

    return features



# =============================================================================
# MANUAL OPS (OPTIONAL ALTERNATIVES TO cv2)
# =============================================================================

def _conv2d_manual(img, kernel):
    # naive convolution using sliding window + reflect padding

    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)

    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            region = padded[y:y + kh, x:x + kw]
            out[y, x] = np.sum(region * kernel, dtype=np.float32)

    return out



def to_gray_manual(bgr):
    # manual rgb->grayscale conversion using luminance weights

    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)

    return 0.114 * b + 0.587 * g + 0.299 * r


def sobel_manual(gray_f):
    # manual sobel edge detection filters

    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    gx = _conv2d_manual(gray_f, kx)
    gy = _conv2d_manual(gray_f, ky)

    return gx, gy

def laplacian_manual(gray_f):
    # manual laplacian operator for edge enhancement

    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=np.float32)

    return _conv2d_manual(gray_f, k)


def gaussian_blur_manual(gray_f, ksize=7, sigma=0.0):
    # gaussian smoothing using generated kernel instead of cv2

    if sigma <= 0.0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8

    ax = np.arange(-ksize // 2 + 1.0, ksize // 2 + 1.0)
    kernel_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()

    kernel_2d = np.outer(kernel_1d, kernel_1d).astype(np.float32)

    return _conv2d_manual(gray_f, kernel_2d)


# =============================================================================
# BUILD TRAINING DATA
# =============================================================================


def build_training_data(n_images, pixels_per_img):

    img_paths = sorted(NYU_IMG_DIR.glob("*.npy"))

    if not img_paths:
        raise FileNotFoundError(
            f"No .npy files found in {NYU_IMG_DIR.resolve()}"
        )

    X_all = []
    y_all = []
    matched_count = 0

    for img_path in img_paths[:n_images]:

        stem = img_path.stem

        number = stem.split("_")[-1]
        gt_path = GT_DIR / f"depth_{number}.npy"

        if not gt_path.exists():
            print(f"[SKIP] Missing GT: {gt_path.name}")
            continue

        print(f"[Training] {img_path.name}")
        matched_count += 1

        bgr = load_npy_as_bgr(img_path)
        gt = np.load(str(gt_path)).astype(np.float32)

        if gt.max() - gt.min() < 1e-8:
            continue

        gt = (gt - gt.min()) / (gt.max() - gt.min())

        features = extract_features(bgr)

        h, w, c = features.shape
        X = features.reshape(-1, c)
        y = gt.reshape(-1)

        idx = np.random.choice(len(X), min(pixels_per_img, len(X)), replace=False)

        X_all.append(X[idx])
        y_all.append(y[idx])

    if len(X_all) == 0:
        raise RuntimeError("no valid image/gt pairs found")

    X_all, y_all = smart_sample(X_all, y_all)

    print(f"\nmatched pairs: {matched_count}")
    print(f"training samples: {len(X_all)}")

    return X_all, y_all

    img_paths = sorted(NYU_IMG_DIR.glob("*.npy"))

    X_all = []
    y_all = []

    for img_path in img_paths[:n_images]:

        if len(X_all) == 0:
            raise RuntimeError("No valid image-GT pairs found")

        # Convert list-of-arrays → single 2D matrix
        X_all = np.concatenate(X_all, axis=0).astype(np.float32)
        y_all = np.concatenate(y_all, axis=0).astype(np.float32)

        # FINAL SAFETY CHECK
        assert X_all.ndim == 2, f"X is not 2D: {X_all.shape}"
        assert y_all.ndim == 1, f"y is not 1D: {y_all.shape}"

        print(f"\nFINAL FLATTENED SHAPE")
        print(f"X: {X_all.shape}")
        print(f"y: {y_all.shape}")

        return X_all, y_all

        # FINAL SAFETY CHECK
        X_all, y_all = smart_sample(X_all, y_all)

        print(f"\nMatched training pairs: {matched_count}")
        print(f"Training samples: {len(X_all)}")

        return X_all, y_all
    print(f"\nTraining samples: {len(X_all)}")

    return X_all, y_all


# =============================================================================
# TRAIN MODEL
# =============================================================================


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib


def train_model(X, y):

    print("\n[XGBoost Training]\n")

    X = np.asarray(X)   # convert input features to numpy array for efficient processing
    y = np.asarray(y)   # convert target depth values to numpy array

    # reshape feature matrix into (num_samples, num_features)
    # each row = pixel-level feature vector, each column = feature type
    X = X.reshape(-1, X.shape[-1])

    # flatten target depth map into 1d vector aligned with X rows
    y = y.reshape(-1)

    # split dataset into training and testing subsets
    # test_size=0.2 means 20% data used for validation
    # random_state ensures reproducibility of split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # xgboost regressor for learning nonlinear mapping from features -> depth
    # n_estimators: number of boosting trees
    # max_depth: controls tree complexity (prevents overfitting)
    # learning_rate: step size shrinkage for stable training
    # subsample: fraction of samples per tree (regularization)
    # colsample_bytree: fraction of features per tree
    # tree_method=hist: faster histogram-based training
    model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        tree_method="hist"
    )

    # train model on pixel-wise feature-depth mapping
    model.fit(X_train, y_train)

    # evaluate model performance using R^2 score on held-out test set
    score = model.score(X_test, y_test)

    print(f"Validation R² Score: {score:.4f}")

    # save trained model to disk for later inference
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved: {MODEL_PATH}")

    return model


# =============================================================================
# LOAD MODEL
# =============================================================================


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run training first."
        )

    print(f"Loading model: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


# =============================================================================
# DEPTH PREDICTION
# =============================================================================


def predict_depth(bgr, model):

    # extract handcrafted + gradient + spatial features from input image
    features = extract_features(bgr)

    h, w, c = features.shape   # image height, width, number of feature channels

    # flatten spatial feature map into pixel-wise feature vectors
    X = features.reshape(-1, c)

    # predict depth value for each pixel independently
    pred = model.predict(X)

    # sanity check to ensure output matches image pixel count
    if len(pred) != h * w:
        raise ValueError("Prediction shape mismatch")

    # reshape flat predictions back into 2d depth map
    depth = pred.reshape(h, w)

    # clamp depth into valid normalized range [0, 1]
    depth = np.clip(depth, 0.0, 1.0)

    # convert to uint8 for visualization purposes
    depth_u8 = normalize_u8(depth)

    return depth.astype(np.float32), depth_u8


# =============================================================================
# PARALLAX
# =============================================================================


def create_parallax(bgr, depth_u8, strength):

    h, w = depth_u8.shape  # image dimensions

    # normalize depth map to [0,1] for displacement scaling
    depth_norm = depth_u8.astype(np.float32) / 255.0

    # compute pixel-wise horizontal shift based on depth
    shift = depth_norm * strength

    # generate coordinate grid for remapping pixels
    gx, gy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    # shift x-coordinates to simulate camera parallax (closer objects move more)
    map_x = np.clip(gx - shift, 0, w - 1)

    # remap image using computed displacement field
    return cv2.remap(
        bgr,
        map_x,
        gy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


# =============================================================================
# METRICS
# =============================================================================


def compute_metrics(pred, gt):

    # convert predictions and ground truth to float for numerical stability
    pred_f = pred.astype(np.float32)
    gt_f   = gt.astype(np.float32)

    # resize prediction if spatial dimensions mismatch ground truth
    if pred_f.shape != gt_f.shape:

        pred_f = cv2.resize(
            pred_f,
            (gt_f.shape[1], gt_f.shape[0])
        )

    # structural similarity index (perceptual similarity metric)
    ssim_val = ssim_fn(
        pred_f,
        gt_f,
        data_range=1.0
    )

    # mean squared error for pixel-wise difference
    mse = np.mean((pred_f - gt_f) ** 2)

    # peak signal-to-noise ratio derived from mse
    psnr_val = (
        10 * np.log10(1.0 / (mse + 1e-9))
    )

    # mean absolute error (robust linear deviation measure)
    mae_val = np.mean(np.abs(pred_f - gt_f))

    # root mean squared error (penalizes large deviations more)
    rmse_val = np.sqrt(mse)

    # spearman correlation for ranking consistency of depth values
    rho, _ = spearmanr(
        pred_f.ravel(),
        gt_f.ravel()
    )

    # return all evaluation metrics as dictionary
    return {
        "SSIM": ssim_val,
        "PSNR": psnr_val,
        "MAE": mae_val,
        "RMSE": rmse_val,
        "Spearman": rho
    }


# =============================================================================
# METRIC PRINTING
# =============================================================================


def print_metrics(stem, m):

    print(f"\n{'─'*60}")

    print(f"  METRICS — {stem}")

    print(f"{'─'*60}")

    print(f"  SSIM       : {m['SSIM']:.4f}")
    print(f"  PSNR       : {m['PSNR']:.2f}")
    print(f"  MAE        : {m['MAE']:.4f}")
    print(f"  RMSE       : {m['RMSE']:.4f}")
    print(f"  Spearman ρ : {m['Spearman']:.4f}")

    print(f"{'─'*60}\n")


# =============================================================================
# OUTPUT SUMMARY
# =============================================================================


def print_output_summary(stem,
                         depth_u8,
                         depth_metres=None,
                         parallax=None):

    print(f"\n{'─'*65}")
    print(f"  OUTPUT ANALYSIS — {stem}")
    print(f"{'─'*65}")

    # compute average intensity of predicted depth map
    mean_depth = np.mean(depth_u8)

    # measure variation in predicted depth distribution
    std_depth  = np.std(depth_u8)

    print(f"  Mean Depth Intensity : {mean_depth:.2f}")
    print(f"  Std Depth            : {std_depth:.2f}")

    # estimate proportion of near vs far regions in scene
    near_pixels = np.sum(depth_u8 > 180)
    far_pixels  = np.sum(depth_u8 < 70)

    total = depth_u8.size

    print(f"  Near Pixels          : {(near_pixels/total)*100:.2f}%")  # close objects
    print(f"  Far Pixels           : {(far_pixels/total)*100:.2f}%")    # distant objects

    # edge detection on depth map to analyze structural transitions
    edges = cv2.Canny(depth_u8, 100, 200)

    edge_density = np.mean(edges > 0) * 100.0

    print(f"  Edge Density         : {edge_density:.2f}%")  # texture/structure complexity

    if parallax is not None:
        print(f"  Parallax Generated   : YES")  # indicates 3d effect was created

    if depth_metres is not None:

        print(f"  Mean Depth (float)   : {np.mean(depth_metres):.4f}")  # raw predicted depth
        print(f"  Max Depth (float)    : {np.max(depth_metres):.4f}")   # farthest estimated point

    print(f"{'─'*65}\n")


# =============================================================================
# MAIN
# =============================================================================


def run_train():

    X, y = build_training_data(
        N_TRAIN_IMAGES,
        PIXELS_PER_IMG
    )

    train_model(X, y)


def run_predict(model=None):

    if model is None:
        model = load_model()

    all_img_paths = sorted(NYU_IMG_DIR.glob("*.npy"))

    n_total = len(all_img_paths)

    test_paths = all_img_paths[int(n_total * 0.8):]

    if not test_paths:
        test_paths = all_img_paths

    if not test_paths:
        raise FileNotFoundError(
            f"No .npy images found in {NYU_IMG_DIR.resolve()}"
        )

    all_metrics = []

    for img_path in test_paths:

        print(f"\n[Predicting] {img_path.name}")

        bgr = load_npy_as_bgr(img_path)

        if bgr is None:
            print("  [SKIP] unreadable")
            continue

        depth_metres, depth_u8 = predict_depth(
            bgr,
            model
        )

        stem = img_path.stem

        # Save depth
        out_npy = OUTPUT_DIR / f"{stem}.npy"

        np.save(str(out_npy), depth_metres)

        print(f"  Depth .npy : {out_npy}")

        out_jpg = OUTPUT_DIR / f"{stem}_vis.jpg"

        cv2.imwrite(str(out_jpg), depth_u8)

        print(f"  Depth vis  : {out_jpg}")

        # Colormap
        if SAVE_COLORMAP:

            cmap = cv2.applyColorMap(
                depth_u8,
                cv2.COLORMAP_INFERNO
            )

            color_path = OUTPUT_DIR / f"{stem}_color.jpg"

            cv2.imwrite(str(color_path), cmap)

            print(f"  Colormap   : {color_path}")

        # Parallax
        par = None

        if CREATE_PARALLAX:

            par = create_parallax(
                bgr,
                depth_u8,
                PARALLAX_STR
            )

            par_path = OUTPUT_DIR / f"{stem}_parallax.jpg"

            cv2.imwrite(str(par_path), par)

            print(f"  Parallax   : {par_path}")

        # Our output summary
        print_output_summary(
            stem,
            depth_u8,
            depth_metres,
            par
        )

        # GT metrics
        gt = load_gt(stem)

        if gt is not None:

            m = compute_metrics(
                depth_metres,
                gt
            )

            all_metrics.append((stem, m))

            print_metrics(stem, m)

        else:
            print("  No GT found.")

        # Preview
        if SHOW_PREVIEW:

            cv2.imshow("Original", bgr)

            cv2.imshow(
                "Depth",
                cv2.applyColorMap(
                    depth_u8,
                    cv2.COLORMAP_INFERNO
                )
            )

            if CREATE_PARALLAX:
                cv2.imshow("Parallax", par)

            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                break

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    # Summary
    if all_metrics:

        print(f"\n{'═'*75}")
        print(f"  FINAL SUMMARY")
        print(f"{'═'*75}")

        for s, m in all_metrics:

            print(
                f"{s:<15}"
                f"SSIM={m['SSIM']:.4f}   "
                f"PSNR={m['PSNR']:.2f}   "
                f"MAE={m['MAE']:.4f}   "
                f"RMSE={m['RMSE']:.4f}"
            )

        print(f"{'═'*75}\n")

    else:

        print(f"\nOutputs saved to {OUTPUT_DIR}/\n")


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="RF Depth Estimator"
    )

    parser.add_argument(
        "--mode",
        choices=["train", "predict", "both"],
        default="both"
    )

    args = parser.parse_args()

    if args.mode == "train":

        run_train()

    elif args.mode == "predict":

        run_predict()

    else:

        X, y = build_training_data(N_TRAIN_IMAGES, PIXELS_PER_IMG)

        print("Final X shape:", X.shape)
        print("Final y shape:", y.shape)

        model = train_model(X, y)

        run_predict(model)