# ============================================================
# AI DEPTH STUDIO — STREAMLIT APP (REDESIGNED UI)
# ============================================================

import streamlit as st
import numpy as np
import cv2
import joblib
import time
from streamlit_image_comparison import image_comparison
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

# =========================
# CONFIG
# =========================

MODEL_PATH = "rf_depth_model.pkl"
GT_DIR = Path("groundTruths")

st.set_page_config(
    page_title="Depth Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# =========================
# FEATURE PIPELINE (MUST MATCH TRAINING)
# =========================

# extract handcrafted features used during training
def extract_features(bgr):

    # convert image to grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # normalize grayscale values to 0-1
    gray = gray.astype(np.float32) / 255.0

    # sobel edge detection using opencv
    # computes horizontal + vertical gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # gradient magnitude map
    grad = np.sqrt(gx**2 + gy**2)

    # laplacian operator for texture + edge intensity
    lap = cv2.Laplacian(gray, cv2.CV_32F)

    # gaussian blur for local mean intensity
    mean = cv2.GaussianBlur(gray, (7,7), 0)

    # local variance feature
    variance = (gray - mean) ** 2

    # normalized pixel coordinate features
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]

    xx = xx.astype(np.float32) / w
    yy = yy.astype(np.float32) / h

    # combine all features into feature tensor
    features = np.stack([
        gray,
        grad,
        np.abs(lap),
        variance,
        xx,
        yy
    ], axis=-1)

    return features

# =========================
# MANUAL OPS (OPTIONAL ALTERNATIVES TO cv2)
# =========================

def _conv2d_manual(img, kernel):

    kh, kw = kernel.shape

    # compute reflect padding size
    pad_h = kh // 2
    pad_w = kw // 2

    # add reflect padding around image borders
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    # output feature map
    out = np.zeros_like(img, dtype=np.float32)

    for y in range(out.shape[0]):
        for x in range(out.shape[1]):

            # local image region
            region = padded[y:y + kh, x:x + kw]

            # convolution operation
            out[y, x] = np.sum(region * kernel, dtype=np.float32)

    return out

#manual rgb to gray scale conversion 
def to_gray_manual(bgr):
    # Manual BGR->gray conversion; returns float32 in [0, 255] if input is uint8.
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    return 0.114 * b + 0.587 * g + 0.299 * r

#edge detection w sobel 
def sobel_manual(gray_f):
    kx = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=np.float32
    )
    ky = np.array(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        dtype=np.float32
    )

    gx = _conv2d_manual(gray_f, kx)
    gy = _conv2d_manual(gray_f, ky)

    return gx, gy


def laplacian_manual(gray_f):
    #laplacian edge detection (use kernel)
    k = np.array(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
        dtype=np.float32
    )
    return _conv2d_manual(gray_f, k)


def gaussian_blur_manual(gray_f, ksize=7, sigma=0.0):
    # gaussian smoothing (noise reduction)
    if sigma <= 0.0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    ax = np.arange(-ksize // 2 + 1.0, ksize // 2 + 1.0)

    #build 1d kernel, normalize, build 2d kernel, do convol
    kernel_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    kernel_2d = np.outer(kernel_1d, kernel_1d).astype(np.float32)
    return _conv2d_manual(gray_f, kernel_2d)

# =========================
# PREDICTION
# =========================

# run random forest depth prediction
def predict_depth(bgr):

    # extract handcrafted features
    features = extract_features(bgr)

    # flatten features into 2d matrix for sklearn
    h, w, c = features.shape
    X = features.reshape(-1, c)

    pred = model.predict(X)

    # reshape predictions back into image format
    depth = pred.reshape(h, w)

    depth = np.clip(depth, 0, 1)

    # convert depth map to uint8 for visualization
    depth_u8 = (depth * 255).astype(np.uint8)

    return depth, depth_u8

# =========================
# GT LOADER
# =========================

# load matching ground truth depth map
def load_gt(upload_name):

    stem = Path(upload_name).stem
    num = stem.split("_")[-1]

    gt_path = GT_DIR / f"depth_{num}.npy"

    if not gt_path.exists():
        return None

    # load gt depth map
    gt = np.load(gt_path).astype(np.float32)

    # normalize gt map to 0-1
    gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-9)

    return gt

# =========================
# METRICS
# =========================

def compute_metrics(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()

    mae = mean_absolute_error(gt, pred)
    rmse = np.sqrt(mean_squared_error(gt, pred))
    #spearman rank correlation
    rho, _ = spearmanr(pred, gt)

    return mae, rmse, rho

# =========================
# PARALLAX
# =========================

# create parallax effect using predicted depth
def create_parallax(img, depth_u8, strength=18):

    h, w = depth_u8.shape

    # normalize depth map
    depth = depth_u8.astype(np.float32) / 255.0

    # compute horizontal shift amount
    shift = depth * strength

    # generate pixel coordinate grid
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))

    # shift pixels according to depth
    map_x = np.clip(gx - shift, 0, w - 1).astype(np.float32)
    map_y = gy.astype(np.float32)

    # remap image pixels to generate parallax
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

# =========================
# GLOBAL STYLES
# =========================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background-color: #0d0d0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,102,241,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(16,185,129,0.08) 0%, transparent 60%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111114 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    padding-top: 2rem;
}

[data-testid="stSidebar"] * {
    color: #c4c4cf !important;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 20px;
    color: #ffffff !important;
    letter-spacing: -0.5px;
    padding: 0 1rem 2rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

.sidebar-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #10b981);
    display: inline-block;
    flex-shrink: 0;
}

/* ── Sidebar label ── */
.sidebar-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4b4b5a !important;
    padding: 0 1rem;
    margin-bottom: 0.4rem;
    margin-top: 1.5rem;
}

/* ── Radio buttons ── */
[data-testid="stSidebar"] .stRadio label {
    font-size: 14px !important;
    font-weight: 400 !important;
    color: #9a9aaf !important;
    padding: 8px 12px !important;
    border-radius: 8px;
    transition: background 0.15s;
    display: block;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.04) !important;
    color: #e0e0e8 !important;
}

[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
[data-testid="stSidebar"] .stRadio input:checked + label {
    color: #ffffff !important;
    background: rgba(99,102,241,0.15) !important;
}

/* ── Slider ── */
[data-testid="stSidebar"] .stSlider > div > div {
    background: rgba(99,102,241,0.3) !important;
}

[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
    background: #6366f1 !important;
    border: 2px solid #818cf8 !important;
}

/* ── Main page header ── */
.page-header {
    padding: 3rem 0 2.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 2.5rem;
}

.page-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 36px;
    color: #f0f0f6;
    letter-spacing: -1.5px;
    line-height: 1;
    margin: 0;
}

.page-title span {
    background: linear-gradient(90deg, #818cf8 0%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.page-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 300;
    color: #4b4b5a;
    margin-top: 6px;
    letter-spacing: 0.02em;
}

/* ── Section heading ── */
.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4b4b5a;
    margin-bottom: 1rem;
    margin-top: 2rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.05);
}

/* ── Image card ── */
.img-card-label {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4b4b5a;
    margin-bottom: 8px;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.4rem !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #4b4b5a !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #e8e8f0 !important;
    letter-spacing: -0.5px !important;
}

/* ── Uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(99,102,241,0.3) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.6) !important;
}

[data-testid="stFileUploader"] * {
    color: #6b6b80 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] * {
    color: #818cf8 !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #6366f1, #10b981) !important;
}

/* ── Success / info / warning ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
}

.stSuccess {
    background: rgba(16, 185, 129, 0.08) !important;
    color: #34d399 !important;
}

.stInfo {
    background: rgba(99,102,241,0.08) !important;
    color: #818cf8 !important;
}

.stWarning {
    background: rgba(245,158,11,0.08) !important;
    color: #fbbf24 !important;
}

/* ── Image captions ── */
[data-testid="caption"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important;
    color: #3a3a4a !important;
    text-align: center;
    margin-top: 4px;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.05) !important;
    margin: 2rem 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 4px; }

/* ── Hide streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="sidebar-dot"></span>
        Depth Studio
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">View</div>', unsafe_allow_html=True)
    mode = st.radio(
        label="",
        options=["Upload", "Camera", "Compare", "Analysis"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-label">Parallax</div>', unsafe_allow_html=True)
    strength = st.slider("Strength", 5, 40, 18, label_visibility="collapsed")

    if mode != "Camera":
        st.markdown('<div class="sidebar-label">Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        camera_image = None
    else:
        uploaded = None
        st.markdown('<div class="sidebar-label">Camera</div>', unsafe_allow_html=True)
        camera_image = st.camera_input("Take a photo", label_visibility="collapsed")

# =========================
# PAGE HEADER
# =========================

st.markdown("""
<div class="page-header">
    <div class="page-title">Monocular <span>Depth Estimation</span></div>
    <div class="page-subtitle">Random Forest · Ground Truth Evaluation · Parallax Synthesis</div>
</div>
""", unsafe_allow_html=True)

# =========================
# EMPTY STATE
# =========================

image_source = camera_image if mode == "Camera" else uploaded

if image_source is None:
    icon = "📷" if mode == "Camera" else "↖"
    msg = "Take a photo to begin" if mode == "Camera" else "Upload an image from the sidebar to begin"
    
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 6rem 2rem;
        color: #2e2e3e;
        font-family: 'DM Sans', sans-serif;
    ">
        <div style="font-size: 48px; margin-bottom: 16px; opacity: 0.3;">{icon}</div>
        <div style="font-size: 15px; font-weight: 400;">{msg}</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# =========================
# IMAGE LOAD
# =========================

file_bytes = np.frombuffer(image_source.read(), np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if img is None:
    st.error("Could not decode image — please upload a valid JPG or PNG.")
    st.stop()
# convert bgr (opencv) to rgb (streamlit display)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# =========================
# INFERENCE
# =========================

with st.spinner("Running depth model..."):
    time.sleep(0.5)
    depth, depth_u8 = predict_depth(img)

#convert depth map to colored heatmap and add parallax
heatmap = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

parallax = create_parallax(img, depth_u8, strength)
parallax_rgb = cv2.cvtColor(parallax, cv2.COLOR_BGR2RGB)

# =========================
# IMAGE COLUMNS
# =========================

#metric display, matching map loading
st.markdown('<div class="section-heading">Output</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown('<div class="img-card-label">Original</div>', unsafe_allow_html=True)
    st.image(img_rgb, use_container_width=True)

with col2:
    st.markdown('<div class="img-card-label">Depth Map</div>', unsafe_allow_html=True)
    st.image(heatmap, use_container_width=True)

with col3:
    st.markdown('<div class="img-card-label">Parallax</div>', unsafe_allow_html=True)
    st.image(parallax_rgb, use_container_width=True)

# =========================
# SLIDER COMPARISON
# =========================

st.markdown('<div class="section-heading">Comparison Slider</div>', unsafe_allow_html=True)

image_comparison(
    img1=img_rgb,
    img2=heatmap,
    label1="Original",
    label2="Depth"
)

# =========================
# GT COMPARISON
# =========================

gt = load_gt(uploaded.name) if uploaded is not None else None

if gt is not None:

    #eval runs if ground truth exists (to compare)
    st.markdown('<div class="section-heading">Ground Truth Evaluation</div>', unsafe_allow_html=True)

    gt_u8 = (gt * 255).astype(np.uint8)
    mae, rmse, rho = compute_metrics(depth, gt)

    #
    # compute evaluation metrics between prediction and ground truth
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.metric("MAE", f"{mae:.4f}")
    c2.metric("RMSE", f"{rmse:.4f}")
    c3.metric("Spearman ρ", f"{rho:.4f}")

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    colA, colB, colC = st.columns(3, gap="medium")

    with colA:
        st.markdown('<div class="img-card-label">Original</div>', unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)

    with colB:
        st.markdown('<div class="img-card-label">Prediction</div>', unsafe_allow_html=True)
        st.image(heatmap, use_container_width=True)

    with colC:
        st.markdown('<div class="img-card-label">Ground Truth</div>', unsafe_allow_html=True)
        gt_color = cv2.applyColorMap(gt_u8, cv2.COLORMAP_INFERNO)
        st.image(cv2.cvtColor(gt_color, cv2.COLOR_BGR2RGB), use_container_width=True)

else:
    st.warning("No ground truth found for this image.")

# =========================
# MODE-SPECIFIC VIEWS
# =========================

if mode == "Analysis":

    st.markdown('<div class="section-heading">Depth Statistics</div>', unsafe_allow_html=True)

    #basic depth stats and visuals
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.metric("Mean Depth", f"{np.mean(depth):.3f}")
    c2.metric("Max Depth", f"{np.max(depth):.3f}")
    c3.metric("Min Depth", f"{np.min(depth):.3f}")

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    st.progress(float(np.mean(depth)))

elif mode == "Compare":

    st.markdown('<div class="section-heading">Side-by-Side View</div>', unsafe_allow_html=True)

    image_comparison(
        img1=img_rgb,
        img2=heatmap,
        label1="Original",
        label2="Depth"
    )

elif mode in ["Upload", "Camera"]:

    st.success("Processing complete.")