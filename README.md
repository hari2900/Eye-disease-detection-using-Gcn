# 👁️ Eye Disease Detection using GCN and EfficientNet

A deep learning project that detects and classifies common eye diseases—**Cataract**, **Glaucoma**, **Diabetic Retinopathy**, and **Normal**—from retinal fundus images. This system combines **EfficientNet** for feature extraction and **Graph Convolutional Networks (GCNs)** for relational learning on image features, leveraging spatial dependencies between samples for improved diagnostic accuracy.

 Utilized graph-based relationships and spatial dependencies in image features, applied graph coarsening techniques, and achieved 94.69% accuracy with the base GCN model and 95.11% with the refined GCN architecture
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

