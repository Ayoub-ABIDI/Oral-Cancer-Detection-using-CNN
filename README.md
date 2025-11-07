<<<<<<< HEAD
# Oral Cancer Detection using CNN

An intelligent deep learning system that detects **oral cancer** from mouth images using **Convolutional Neural Networks (CNNs)**.  
This project performs **real-time image classification** using a trained `.keras` model integrated in a **Streamlit web interface**.

---

## Project Structure

OralCancerDetection/
├── 📁 .ipynb_checkpoints/ # Jupyter autosave files
├── 📁 DataSet/ # Dataset of oral images (healthy / cancerous)
├── 📁 venv/ # Virtual environment (ignored in Git)
├── 📄 App.py # Streamlit app for real-time detection
├── 📄 Detection.ipynb # Model training and testing notebook
├── 📄 Model.ipynb # Model building and tuning
├── 📄 my_image_classifier.keras # Trained CNN model file
├── 📄 style.css # UI styling for the app
├── 📄 Untitled.ipynb # Experimental or test notebook
└── 📄 README.md # Documentation (this file)

yaml
Copier le code

---

## Project Description

This system detects **oral cancer** from mouth images using a CNN trained on a **Kaggle dataset**.  
The dataset was **imbalanced**, so augmentation techniques were used to create balanced samples.  
Once trained, the model can classify live camera input in real time as:

- ✅ **Healthy**
- ⚠️ **Possible Oral Cancer**

---

## Model Details

- **Architecture:** CNN (Convolutional Neural Network)
- **Layers:** Conv2D → MaxPooling → Dropout → Flatten → Dense
- **Input Size:** 128×128 RGB images
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy, Precision, Recall, F1-score
- **Saved Model:** `my_image_classifier.keras`

---

## Dataset

**Source:** Kaggle – Oral Cancer Image Dataset  
**Classes:**  
- `0`: Healthy Tissue  
- `1`: Cancerous Tissue  

**Preprocessing:**
- Image resizing to 128×128  
- Normalization (values between 0–1)  
- Augmentation for minority class: rotation, flipping, zoom, brightness adjustment  

---

## Training Process

Performed in Jupyter Notebook (`Detection.ipynb` or `Model.ipynb`):

```bash
jupyter notebook Detection.ipynb
Training includes:

Early stopping

Learning rate reduction on plateau

Automatic saving of best model → my_image_classifier.keras

Evaluation
After training, performance was measured using:

Accuracy

Confusion matrix

Precision, Recall, F1-score

ROC curve
=======
# Oral-Cancer-Detection-using-CNN
This system detects oral cancer from mouth images using a CNN
>>>>>>> 3073f595a0fcce4b0ad2d095f4b82657a09e0543

