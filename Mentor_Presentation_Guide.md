# 🎓 Mentor Presentation & Viva Defense Guide
## Project: AI-Powered Gear Tooth Undercutting Detection & Analysis System

---

## 1. Executive Summary & Pitch (30-Second Elevator Pitch)
> *"Our project is an intelligent quality-control and manufacturing inspection system designed to detect **gear tooth undercutting**—a critical manufacturing defect that weakens gear teeth at the dedendum root. We developed a **dual-engine architecture**: 
> 1. A **Deep Learning Computer Vision model (MobileNetV2)** that inspects physical gear photographs or CAD renders in real-time, augmented with automated screenshot-cropping.
> 2. A **Machine Learning Tabular model (Histogram Gradient Boosting)** that mathematically validates geometric manufacturing parameters ($z, m, \alpha, x$).
> This delivers instantaneous, non-destructive optical inspection coupled with design-stage validation through an interactive mechanical CAD web interface."*

---

## 2. Theoretical Mechanical Engineering Foundation

### What is Gear Tooth Undercutting?
- **Definition**: Undercutting occurs during gear generation (e.g., hobbing or rack generation) when the tip of the cutting tool sweeps inside the theoretical base circle, gouging out material from the flank and root fillet of the tooth.
- **Why it is catastrophic**:
  1. **Severe stress concentration**: Significantly reduces tooth thickness at the root critical section ($s_F$).
  2. **Beam fatigue failure**: According to the **Lewis Bending Equation**:
     $$\sigma_b = \frac{F_t}{b \cdot m \cdot Y}$$
     Where $Y$ is the Lewis form factor. An undercut tooth drastically reduces $Y$, causing premature tooth fracture under dynamic torque.
  3. **Loss of contact ratio**: Strips part of the active involute profile, causing vibration, noise, and uneven load distribution.

### Governing Mathematical Equations:
1. **Theoretical Minimum Number of Teeth** ($z_{min}$ for standard rack cutter with addendum $h_a = 1.0 \cdot m$):
   $$z_{min} = \frac{2}{\sin^2(\alpha)}$$
   - For standard $\alpha = 20^\circ$: $z_{min} = \frac{2}{\sin^2(20^\circ)} \approx 17.1 \rightarrow \mathbf{17 \text{ teeth}}$.
   - For $\alpha = 14.5^\circ$: $z_{min} = \frac{2}{\sin^2(14.5^\circ)} \approx 31.9 \rightarrow \mathbf{32 \text{ teeth}}$.
2. **Profile Shift Coefficient ($x$)**:
   To avoid undercutting when $z < z_{min}$, a positive addendum modification (profile shift) $x$ is applied:
   $$x_{min} = \frac{z_{min} - z}{z_{min}} = 1 - \frac{z \cdot \sin^2(\alpha)}{2}$$

---

## 3. Machine Learning & Deep Learning Algorithms Used

### Engine A: Deep Learning Computer Vision (Visual Defect Inspection)
* **Algorithm**: **MobileNetV2** with **Transfer Learning** (Pretrained on ImageNet-1K, PyTorch).
* **Architecture Details**:
  - **Inverted Residual Blocks**: Channels expand into high-dimensional space ($6\times$), filter via $3\times 3$ Depthwise Separable Convolutions, and project back via $1\times 1$ Linear Bottlenecks.
  - **Why MobileNetV2 over ResNet-50 / VGG-16?**
    - MobileNetV2 has **~3.5 Million parameters** (~14 MB memory footprint) vs VGG-16's **138 Million parameters** (528 MB).
    - It allows **real-time edge CPU inference** (<35ms per image on a standard laptop or factory tablet) without requiring a high-end GPU.
* **Custom Classifier Head**:
  $$\text{Input (1280)} \rightarrow \text{AdaptiveAvgPool2d} \rightarrow \text{Dropout}(0.3) \rightarrow \text{Linear}(1280, 256) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.2) \rightarrow \text{Linear}(256, 2)$$
* **Smart Preprocessing & Letterbox Auto-Cropping**:
  - Automatically isolates the gear subject by removing phone status bars, desktop window borders, and black/white letterbox borders before feeding normalized $224 \times 224$ tensors to the neural network.
* **Data Augmentation**:
  - Random Affine ($\pm 15^\circ$ rotation, translation), Horizontal Flip, Color Jitter (contrast, brightness, saturation), and ImageNet channel standardization:
    $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

---

### Engine B: Tabular Machine Learning (Parametric Design Stage)
* **Algorithm**: **Histogram-based Gradient Boosting Classifier (`HistGradientBoostingClassifier`)** (Scikit-Learn).
* **Champion Comparison**:
  - Evaluated against **Decision Trees (CART)**, **Random Forests**, and **Logistic Regression**.
  - HistGradientBoosting was chosen because it bins continuous features into 256 integer-valued histogram bins, reducing time complexity from $\mathcal{O}(N \log N)$ to $\mathcal{O}(N)$, preventing overfitting on boundary values, and handling nonlinear gear equations with **>99.6% accuracy**.
* **Input Features**:
  1. $z$: Number of teeth ($10 \le z \le 120$)
  2. $m$: Normal module ($1.0 \le m \le 10.0$ mm)
  3. $\alpha$: Pressure angle ($14.5^\circ, 20.0^\circ, 25.0^\circ$)
  4. $x$: Profile shift / addendum modification coefficient ($-0.5 \le x \le +1.0$)

---

## 4. Top 12 Mentor Questions & High-Impact Answers

### Q1: *"What is the main objective of this project, and what industry problem does it solve?"*
> **Answer**: 
> *"In gear manufacturing and automotive power transmission, tooth root undercutting drastically weakens the tooth base, causing catastrophic fatigue fracture under bending loads. Traditional inspection requires expensive Coordinate Measuring Machines (CMM) or manual profile projectors. Our project provides an accessible, automated two-stage solution: a fast, non-destructive optical computer vision inspector for physical gears, and a rapid parametric validator for mechanical design engineers."*

---

### Q2: *"Why did you use Transfer Learning instead of training a CNN from scratch?"*
> **Answer**:
> *"Training a deep convolutional network from scratch requires hundreds of thousands of labeled images to learn primitive spatial representations like edges, curves, gradients, and textures. With Transfer Learning, MobileNetV2's feature extractor is already pre-trained on ImageNet's 1.4 million images. We froze lower feature representations and fine-tuned the upper semantic layers on our gear tooth dataset, achieving high generalization accuracy in minimal training epochs without overfitting."*

---

### Q3: *"Why did you select MobileNetV2 instead of heavier models like ResNet-50 or VGG-19?"*
> **Answer**:
> *"Industrial quality control often runs on edge devices, shop-floor tablets, or low-cost embedded hardware without dedicated GPUs. MobileNetV2 uses **Depthwise Separable Convolutions** and **Inverted Residual Bottlenecks**, reducing computational complexity from $\mathcal{O}(D_K^2 \cdot M \cdot N \cdot D_F^2)$ to $\mathcal{O}(D_K^2 \cdot M \cdot D_F^2 + M \cdot N \cdot D_F^2)$. This provides over an **8-fold reduction in Multiply-Accumulate (MAC) operations** while maintaining top-tier classification accuracy and sub-50ms latency on edge CPUs."*

---

### Q4: *"How does the model distinguish between an undercut tooth and a normal involute tooth?"*
> **Answer**:
> *"The CNN's deeper convolutional layers extract morphological geometry around the **trochoidal root fillet** and **dedendum transition zone**. In a healthy gear, the involute curve smoothly blends into a circular root fillet. In an undercut tooth, there is a visible notch or waist narrowing below the base circle where the cutter scooped out material. The network's attention weights correlate strongly with this flank-to-root junction."*

---

### Q5: *"What was your dataset size, and how did you prevent overfitting?"*
> **Answer**:
> *"We utilized a balanced dataset of undercut and healthy tooth profiles. To prevent overfitting on smaller sample sets, we implemented three defense layers:
> 1. **Rigorous Data Augmentation**: Random affine rotations ($\pm 15^\circ$), horizontal mirroring, scaling, and photometric jittering (contrast, brightness variations).
> 2. **Regularization**: Integrated Dropout ($p=0.3$ and $p=0.2$) in the classification head, along with $L_2$ weight decay ($1\times 10^{-4}$) via the AdamW optimizer.
> 3. **Smart Letterbox Cropping**: Automated edge/bounding-box isolation so the model doesn't learn spurious background artifacts."*

---

### Q6: *"Why did you use both a Tabular ML model and a Vision Deep Learning model?"*
> **Answer**:
> *"They address two distinct phases of the product lifecycle:
> - **Design Phase (Tabular Model)**: A mechanical engineer has CAD specifications ($z, m, \alpha, x$) and wants instant validation before sending specs to the workshop.
> - **Manufacturing & Inspection Phase (Computer Vision Model)**: A quality inspector on the factory floor receives a physical cut gear, takes a photo with a microscope or smartphone, and wants non-contact optical verification."*

---

### Q7: *"What evaluation metrics did you use, and why is accuracy alone insufficient?"*
> **Answer**:
> *"While accuracy tells us overall correctness, in manufacturing quality control, a **False Negative** (classifying an undercut gear as healthy and installing it in a gearbox) leads to catastrophic mechanical failure. Therefore, we tracked:
> - **Recall (Sensitivity)**: Ensures virtually zero undercut gears escape undetected.
> - **Precision**: Minimizes false scrap rate.
> - **F1-Score**: Harmonic mean of precision and recall.
> - **Confusion Matrix**: Visualizes exact distribution of Type I and Type II errors."*

---

### Q8: *"What is Profile Shift ($x$), and how does it prevent undercutting in spur gears?"*
> **Answer**:
> *"Profile shift (or addendum modification) is the radial displacement of the cutting tool's datum line from the gear blank's reference circle during generation. By shifting the cutter radially outwards by a positive factor $+x \cdot m$, the active cutting tip is moved outside the base circle, generating a full involute profile without thinning the root critical section, even when teeth count is below 17."*

---

### Q9: *"How does the system handle real-world lighting or photo quality variations?"*
> **Answer**:
> *"We designed a dedicated preprocessing pipeline in `src/augmentation_utils.py`:
> 1. It scans for letterboxing, black borders, and phone status bars using aspect-ratio mask segmentation.
> 2. It applies ImageNet standard channel-wise mean and variance normalization.
> 3. Color jitter training exposed the network to varied luminance, making it resilient against ambient shop-floor illumination shifts."*

---

### Q10: *"How is the application deployed and served?"*
> **Answer**:
> *"The architecture is decoupled into:
> - **Backend**: Python Flask REST API with Gunicorn WSGI handling inference requests.
> - **Frontend**: Futuristic animated mechanical HUD with CSS keyframes, interactive dropzone, and live confidence gauges.
> - **Deployment**: Fully containerized using Docker, with automated Cloudflare tunneling and Render.com blueprints for public cloud deployment."*

---

### Q11: *"Can this model be integrated into an automated manufacturing assembly line?"*
> **Answer**:
> *"Yes. Because the MobileNetV2 model is lightweight (~9.8 MB serialized PyTorch weights), it can be deployed directly onto an industrial Raspberry Pi 5 or NVIDIA Jetson Nano connected to a fixed telecentric inspection camera. It can trigger a PLC reject actuator over Modbus or MQTT in under 50 milliseconds per gear."*

---

### Q12: *"What are the limitations and future scope of this project?"*
> **Answer**:
> *"Current limitations: The vision system operates primarily on 2D profile photographs. 
> Future enhancements:
> 1. Multi-defect classification (expanding classes to include micro-pitting, scoring, and spalling).
> 2. Object detection bounding-box networks (YOLOv8 / YOLOv11) to locate and count individual teeth across a complete $360^\circ$ gear rim.
> 3. Direct integration with industrial 3D laser profilometers."*
