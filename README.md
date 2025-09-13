# satellite-imagery-downscaling-landsat8-to-sentinel2

**Super-resolution of multispectral satellite imagery** — building a reproducible dataset and deep learning models for downscaling **Landsat-8/9 (30 m)** to **Sentinel-2 (10 m)**.

---

## 📖 Overview

This project focuses on the creation of a **paired dataset** and **deep learning pipeline** for downscaling satellite imagery.
We target **regions without snow or desert coverage**, generate **512×512 patches**, and provide standardized preprocessing steps including:

* Atmospheric correction
* Cloud and shadow masking
* Co-registration between sensors
* Band harmonization

The ultimate goal is to benchmark and analyze different AI models (CNNs, GANs, U-Nets, Transformers) for **super-resolution of multispectral data**.

---

## ✨ Features

* 🛰️ Automated download of Sentinel-2 (10 m) and Landsat-8/9 (30 m) using STAC APIs
* 🌍 Selection of AOIs in regions without snow/desert artifacts
* 🧩 Patch generation (512×512 px at 10 m resolution)
* ☁️ Cloud/shadow quality control with masks
* 🔬 Baseline AI models for downscaling (to be added)
* 📊 Evaluation across different model architectures

---

## 🗂 Dataset Specification

* **Resolution target**: 10 m/pixel (Sentinel-2)
* **Patch size**: 512×512 px (\~5.12×5.12 km²)
* **Temporal pairing**: Landsat-8/9 and Sentinel-2 with Δt ≤ 5 days
* **Cloud coverage**: ≤ 15% per patch (after masking)
* **Bands**:

  * Sentinel-2: B2, B3, B4, B8 (+ optional harmonized 20 m bands)
  * Landsat-8/9: B2, B3, B4, B5, B6, B7 (resampled to 10 m)

Data is not distributed in this repository due to size.
Instead, detailed instructions are provided to **reproduce the dataset**.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/satellite-imagery-downscaling-landsat8-to-sentinel2.git
cd satellite-imagery-downscaling-landsat8-to-sentinel2

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install requirements
pip install -U pip
pip install -r requirements.txt
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

