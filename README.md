# Monocular Depth Estimation - AI Depth Studio

This project is a Digital Image Processing (DIP) application that estimates the depth of objects in a 2D image. It uses classical handcrafted features (like Sobel edge gradients, Laplacian texture operators, and local variance) combined with a machine learning regression model (Random Forest / XGBoost) to infer depth maps directly from single monocular images.

## Features

- **Depth Prediction:** Converts standard RGB images into depth maps using an XGBoost/Random Forest model trained on NYU dataset features.
- **AI Depth Studio:** A stylized Streamlit web interface for an interactive experience.
- **Parallax Generation:** Generates pseudo-3D visual shifts based on the predicted depth to create interactive parallax.
- **Ground Truth Evaluation:** If a ground truth array is available, compares prediction against it using metrics such as SSIM, PSNR, MAE, RMSE, and Spearman ρ.

## Project Structure

```
📁 23i0605_23i0736_23i0751_DIP_B/
├── 📁 app/
│   └── app.py                  # The Streamlit web application
├── 📁 models/
│   ├── rf_depth_model.pkl      # Trained model file
│   └── depth_model.pkl         # Alternate/previous trained model
├── 📁 src/
│   ├── dipProj.py              # Main training and bulk-prediction script
│   ├── dataset_viewer.py       # Helper script for viewing dataset
│   └── read_mat.py             # Helper script to convert .mat files
├── 📁 rawImages/               # Source 2D .npy images (needs to be populated)
├── 📁 groundTruths/            # Ground truth .npy depth arrays
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Make sure you have Python 3.8+ installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Running the Web Application
To start the interactive "Depth Studio" UI:
```bash
streamlit run app/app.py
```
This will open the web application in your browser where you can upload an image or use your webcam to see real-time depth mapping and parallax generation.

### 2. Training the Model
If you want to retrain the model or run bulk predictions over a dataset of `.npy` arrays, you can use the main script:
```bash
python src/dipProj.py
```
*(Make sure the `rawImages` and `groundTruths` directories contain the paired dataset files prior to running the training).*

### Collaborators
This project was developed as a team effort by:
- [Zahra Zaheer](https://github.com/zahra745)
- [Ayesha Noor](https://github.com/Ayesha-Noor-04)
- [Haaniah Ismail](https://github.com/haani-maybe)
