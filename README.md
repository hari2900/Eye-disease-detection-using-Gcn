# 👁️ Eye Disease Detection using GCN and EfficientNet

A deep learning project that detects and classifies common eye diseases—**Cataract**, **Glaucoma**, **Diabetic Retinopathy**, and **Normal**—from retinal fundus images. This system combines **EfficientNet** for feature extraction and **Graph Convolutional Networks (GCNs)** for relational learning on image features, leveraging spatial dependencies between samples for improved diagnostic accuracy.Utilized graph-based relationships and spatial dependencies in image features, applied graph coarsening techniques, and achieved 94.69% accuracy with the base GCN model and 95.11% with the refined GCN architecture
---

## 📌 Features

- EfficientNet-based deep feature extraction from fundus images.
- Graph construction using extracted features to capture similarity.
- GCN-based disease classification.
- Graph coarsening and refinement for optimized learning.
- Comparative performance analysis between base GCN and refined GCN.

---

## 🗂️ Dataset

The model was trained and evaluated on a curated dataset containing labeled retinal images across 4 categories:

- **Cataract**
- **Glaucoma**
- **Diabetic Retinopathy**
- **Normal**

> 📝 Note: "https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification"
---

## 🏗️ Project Structure

```bash
Eye-Disease-Detection-GCN/
├── data/
│ ├── raw/ # Original dataset (4 disease classes)
│ │ ├── diabetic_retinopathy/
│ │ ├── cataract/
│ │ ├── glaucoma/
│ │ └── normal/
│ ├── preprocessed/ # Preprocessed images
│ │ ├── diabetic_retinopathy/
│ │ ├── cataract/
│ │ ├── glaucoma/
│ │ └── normal/
│ └── graph/ # Graph data (EfficientNet features, labels, edges)
│ ├── features.npy
│ ├── labels.npy
│ └── edge_index.npy
├── models/
│ ├── efficientnet_features.py # Extracts and saves features.npy and labels.npy
│ └── gcn_model.py # GCN architecture

---
```

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- scikit-learn
- pandas
- numpy

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Preprocess and Load the Data

```bash
python utils/data_loader.py
```

### 2. Train the GCN Model

```bash
python train_gcn.py
```

### 3. Evaluate the Model

```bash
python evaluate_gcn.py
```

---

## 📊 Model Accuracy Comparison

| Model             | Train Accuracy | Test Accuracy | Test Loss |
|------------------|----------------|---------------|-----------|
| GCN Model         | 94.69%         | 93.01%        | 14.52     |
| Refined GCN Model | 95.11%         | 92.01%        | 13.90     |

---

## 📈 Classification Report

```python
from sklearn.metrics import classification_report

model.eval()
out = model(data.x, data.edge_index)
pred = out.argmax(dim=1).cpu()
print(classification_report(y.cpu(), pred, digits=3))
```

**Output:**

```
              precision    recall  f1-score   support

           0      0.966     0.960     0.963      1038
           1      0.997     0.994     0.995      1098
           2      0.917     0.887     0.902      1007
           3      0.902     0.939     0.920      1074

    accuracy                          0.946      4217
   macro avg      0.945     0.945     0.945      4217
weighted avg      0.946     0.946     0.946      4217
```

---

## ✅ Conclusion

- GCN models are effective in learning from structured image representations using graph data.
- The refined GCN architecture shows improved training performance, though it slightly overfits compared to the base model.
- Precision, recall, and F1-scores across all classes indicate robust and consistent model behavior.

---

## 🧠 Future Work

- 🔬 Explore deeper GCN layers and attention-based architectures like Graph Attention Networks (GAT).
- 📉 Integrate end-to-end learning from raw images to final classification using hybrid models.
- 🔍 Incorporate explainability using **Grad-CAM**, **Score-CAM**, **LRP**, or **SHAP** to visualize disease-relevant features.

---
