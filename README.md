# ⚙️ Gear Tooth Undercutting Detection & Analysis System

An intelligent AI-powered mechanical inspection system designed to detect and prevent **Gear Tooth Undercutting** in spur and helical gears. The system combines **Deep Learning Computer Vision (MobileNetV2)** for visual inspection with **Tabular Machine Learning (HistGradientBoosting)** for geometric engineering parameter validation, delivered through a futuristic, animated mechanical CAD web interface.

---

## 🌟 Key Features

- **Dual-Engine Inspection**:
  1. 📷 **Vision Classifier (Deep Learning)**: Upload photos or CAD renders of gear tooth flanks to detect root undercutting using a fine-tuned MobileNetV2 architecture.
  2. 📐 **Geometric Parameter Classifier (Machine Learning)**: Input mechanical gear parameters (teeth count, module, pressure angle, addendum coefficient) to evaluate manufacturing risk.
- **Smart Letterbox & Screenshot Cropping**:
  - Automatically isolates the gear subject from phone screenshots, desktop screen captures, and black/white letterbox borders before passing it to the neural network.
- **Cyberpunk / CAD Animated Web Interface**:
  - Interactive file dropzone with live preview.
  - Animated HUD scanning overlay with dynamic laser reticle and soundwave animations.
  - Visual confidence gauges, risk level indicators, and detailed engineering recommendations.
- **Ready for Deployment**:
  - Supports local hosting, Docker containerization, and public exposure via Cloudflare Tunnel.

---

## 📁 Repository Structure

```
gear-model/
├── dataset/                        # Image dataset for computer vision
│   ├── undercut/                   # Images showing undercut root profiles
│   └── no_undercut/                # Images of healthy involute tooth profiles
├── models/
│   └── gear_cnn_model.pt           # Trained PyTorch MobileNetV2 model weights
├── src/
│   ├── augmentation_utils.py       # Data augmentation & smart letterbox cropping
│   └── inference.py                # Image preprocessing and PyTorch inference pipeline
├── static/
│   ├── css/
│   │   └── style.css               # Futuristic CAD/mechanical HUD animations & styles
│   └── uploads/                    # Temporary directory for uploaded gear photos
├── templates/
│   └── index.html                  # Responsive web dashboard with dual inspection tabs
├── app.py                          # Flask web application & REST API routes
├── Dockerfile                      # Production Docker container definition
├── gear_undercut_final_model.joblib# Trained tabular HistGradientBoosting model
├── gear_undercutting_ML_dataset.csv# Dataset of mechanical gear parameters
├── requirements.txt                # Python package dependencies
├── train_final_model.py            # Training script for tabular parameter model
├── train_image_classifier.py       # Training script for PyTorch image classifier
├── .gitignore                      # Git ignore rules for virtual environments & uploads
├── LICENSE                         # MIT License
└── README.md                       # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.10+** (Python 3.11 or 3.12 recommended)
- **Git**

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/gear-model.git
cd gear-model
```

### 3. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell / Command Prompt)
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Web Application
```bash
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔬 Model Architectures & Training

### 1. Computer Vision (PyTorch Transfer Learning)
- **Base Architecture**: MobileNetV2 pre-trained on ImageNet.
- **Classifier Head**:
  - Adaptive Average Pooling $\rightarrow$ Dropout(0.3) $\rightarrow$ Linear(1280, 256) $\rightarrow$ ReLU $\rightarrow$ Dropout(0.2) $\rightarrow$ Linear(256, 2).
- **Data Augmentation**: Random affine transforms, color jitter, horizontal flips, and synthetic letterbox cropping.
- **To Retrain**:
  ```bash
  python train_image_classifier.py
  ```
  The best weights will be saved to `models/gear_cnn_model.pt`.

### 2. Tabular Machine Learning (Scikit-Learn)
- **Champion Algorithm**: HistGradientBoostingClassifier.
- **Features Used**:
  - Number of Teeth ($z$)
  - Normal Module ($m$)
  - Pressure Angle ($\alpha$)
  - Addendum Modification Coefficient ($x$)
- **To Retrain**:
  ```bash
  python train_final_model.py
  ```
  The trained pipeline will be serialized to `gear_undercut_final_model.joblib`.

---

## 🐳 Docker Deployment

You can build and run this application inside a lightweight container:

```bash
# Build the Docker image
docker build -t gear-undercut-detector .

# Run the container on port 5000
docker run -p 5000:5000 gear-undercut-detector
```

---

## 🌐 Public Sharing via Cloudflare Tunnel

To make your local server publicly accessible securely without port forwarding:
```bash
cloudflared.exe tunnel --url http://127.0.0.1:5000
```
Copy the generated `*.trycloudflare.com` URL and share it with collaborators.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
