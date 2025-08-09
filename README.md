# 🩺 Covid‑positive AI Model

An AI-powered project to distinguish COVID-19 positive vs. normal chest X-ray scans using deep learning.

---

## ✨ Features

- 🧠 **CNN-based classifier** to detect COVID-positive vs. normal X-rays.
- 🛠 **Training data pipeline** using `ImageDataGenerator` and binary classification.
- 💾 **Saved model** (`covid_xray_model.h5`) for inference.
- 📊 **Evaluation suite** (`evaluate.py`) displaying confusion matrix, ROC, precision–recall curves, sample predictions, and Grad‑CAM visualizations.
- 🔍 **Prediction script** (`predict_covid_xray.py`) for single-image inference.

---

## 📂 Project Structure

```
├── dataset/                      # Organizes images into `covid/` and `normal/` folders
├── covid_xray_model.h5           # Trained model weights
├── model.py                      # CNN model architecture and training pipeline
├── evaluate.py                   # Visual evaluation of model performance
├── predict_covid_xray.py         # Script for predicting individual images
└── README.md                     # Project documentation
```

---

## ⚙️ Installation

1. 📥 Clone the repository:
   ```bash
   git clone https://github.com/joebrashear31/Covid-positive-AI-Model.git
   cd Covid-positive-AI-Model
   ```
2. 📦 Install dependencies:
   ```bash
   pip install tensorflow scikit-learn matplotlib
   ```

---

## 🚀 Usage Examples

### 🏋️‍♂️ Training the Model

Run `model.py` to train your CNN on the dataset:
```bash
python model.py
```
- 📚 Automatically loads images from `dataset/`, builds and trains the CNN, and saves the best model to `covid_xray_model.h5`.

### 📈 Evaluating the Model

Use `evaluate.py` to analyze model performance:
```bash
python evaluate.py
```
Features:
- 📊 Confusion matrix (both raw counts and normalized).
- 📉 ROC curve with AUC score.
- 📏 Precision–recall curve with AUC score.
- 🖼 Visual display of random and misclassified samples (correct in green, wrong in red).
- 🔥 Grad‑CAM overlays to interpret model focus.

### 🧪 Predicting on a Single Image

Use the prediction script to test individual X-rays:
```bash
python predict_covid_xray.py path/to/xray.jpg
```
Outputs:
- 🖼 Displays the image.
- 🏷 Shows model’s prediction (`Normal` vs. `COVID Positive`) with probability.

---

## 🧾 Notes on Labeling Logic

The project uses:
```
{'covid': 0, 'normal': 1}
```
- 📈 **Higher probabilities** → more likely Normal.
- 📉 **Lower probabilities** → more likely COVID.

Scripts reflect this mapping.

---

## 🌟 Future Enhancements

- 📂 Batch predictions and CSV exports.
- 🌐 Deployment-ready API via Flask or FastAPI.
- 🔬 Additional explainability methods (e.g., LIME, SHAP).
- 📊 Fine-tuning with transfer learning for improved accuracy.

---

## 📝 License & Acknowledgment

📜 This project is open-source and available for educational and research use. Contributions and feedback are welcome!

---

## 🙋 About Me

Hi! I’m **Joe Brashear** 👋  
💻 Software engineer turned AI enthusiast.  
🧠 Passionate about deep learning, medical imaging, and building impactful healthcare tools.  
📍 Based in the USA 🇺🇸  
💬 Connect with me on [GitHub](https://github.com/joebrashear31) or drop me a message for collaborations!

---
