# Crop Disease Detection 🚜🌿

A lightweight web application that predicts plant diseases from leaf images using a pre‑trained Convolutional Neural Network (CNN).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://crop-diseasedetection.streamlit.app/#advanced-plant-pathology-engine)

## 🔗 Live Application
Experience the deep learning classifier live in your browser:  
🚀 **[AgriShield Pathology Web App](https://crop-diseasedetection.streamlit.app/#advanced-plant-pathology-engine)**

## 🎯 Goal
Enable farmers and agronomists to quickly identify common crop diseases without expensive lab equipment, helping to reduce yield loss and improve sustainable farming practices.

## 🗂️ Repository Structure
```
Crop_disease_detection/
│
├─ 📄 README.md                # You are here – project overview
├─ 📄 main.py                  # Streamlit UI + inference pipeline
├─ 📁 train/                   # Training scripts & notebooks
│   └─ *.ipynb
├─ 📁 test/                    # Evaluation notebooks
│   └─ *.ipynb
├─ 📁 valid/                   # Validation data
├─ 📄 class_name.json          # Mapping of class indices to disease names
├─ 📄 training_hist.json       # Training history (loss/accuracy)
├─ 📄 trained_plant_disease_model.keras  # **Model weights (tracked with Git LFS)**
└─ .gitattributes             # LFS configuration for *.keras files
```

## ⚙️ Installation
```bash
# Clone the repo
git clone https://github.com/manishswami1114/Crop_Disease_Detection.git
cd Crop_Disease_Detection

# Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt
```

*If you encounter large‑file errors, ensure **Git LFS** is installed:*  
```bash
brew install git-lfs   # macOS
git lfs install
```

## 🚀 Running the App
```bash
streamlit run main.py
```
Open the displayed URL (by default `http://localhost:8501`) and upload a leaf image. The app will display the predicted disease along with a visualisation.

## 📚 Model Details
- Architecture: Custom CNN (see `train/` notebooks for the exact layers)
- Input size: `128 × 128 × 3`
- Trained on a curated dataset of healthy and diseased leaf images.
- Model file is stored with **Git LFS** to keep the repository lightweight.

## 🤝 Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b <feature-name>`).
3. Make your changes and ensure they pass existing notebooks.
4. Submit a Pull Request.

## 📄 License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---
*Built with ❤️ by the Crop Disease Detection team.*
