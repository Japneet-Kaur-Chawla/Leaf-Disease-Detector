# 🌿 Plant Disease Detection System

An end-to-end Machine Learning pipeline and interactive web application designed to identify plant diseases from leaf images. 

This project extracts deep-learning features from leaf images using **EfficientNet-B0 (PyTorch)**, optimizes feature selection using a custom **Genetic Algorithm (GA)**, and classifies the diseases using highly tuned **Random Forest** and **SVM** models. The user interface is built with **Streamlit** for a seamless, interactive experience.

---

## ✨ Features
* **Deep Feature Extraction:** Uses a pre-trained EfficientNet-B0 model to extract 1280 robust features per image.
* **Evolutionary Feature Selection:** Implements a Genetic Algorithm (GA) to find the most optimal subset of features, reducing dimensionality while maintaining high accuracy.
* **Interactive UI:** A beautiful, theme-aware (Light/Dark mode) Streamlit frontend that provides real-time predictions, disease information, and treatment guidance.
* **Caching & Optimization:** Utilizes Streamlit caching to load heavy ML models into memory only once, ensuring lightning-fast predictions.

---

## 📂 Project Structure

```text
├── app.py                     # Streamlit frontend application
├── main.py                    # Main pipeline script (trains the models)
├── preprocessing.py           # Handles dataset splitting (train/val/test)
├── feature_extraction.py      # PyTorch EfficientNet feature extraction
├── ga_feature_selection.py    # Genetic Algorithm for feature optimization
├── mlmodel_training.py        # RF and SVM training & cross-validation scripts
├── requirements.txt           # Python dependencies
└── results/                   # (Ignored by Git) Folder containing trained .pkl and .npz models



### ⚙️ Step-by-Step Instructions to Run This Project

If you are an interviewer or a developer testing this app, follow these exact terminal commands to run the project on your local machine.

### Step 1: Download the Code
Open your terminal and clone this repository to your computer:
```bash
git clone [https://github.com/Japneet-Kaur-Chawla/Leaf-Disease-Detector.git](https://github.com/Japneet-Kaur-Chawla/Leaf-Disease-Detector.git)

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download the pretrained models
```bash
python download_models.py
```
This downloads 4 files into a `results/` folder:
- `PlantVillage_selected_rf.pkl`
- `PlantVillage_selected_svm.pkl`
- `PlantVillage_ga_mask.npz`
- `PlantVillage_features_train.npz`

Alternatively, download them manually from [this Google Drive folder](YOUR_LINK_HERE) and place them in a `results/` folder in the project root.

### Step 4: Run the app
```bash
streamlit run app.py
```
This opens the app in your browser at `http://localhost:8501`. Upload a leaf image and click **Predict**.

---


