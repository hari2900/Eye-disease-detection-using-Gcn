# 👁️ Eye Disease Detection using GCN and EfficientNet

A deep learning project that detects and classifies common eye diseases—**Cataract**, **Glaucoma**, **Diabetic Retinopathy**, and **Normal**—from retinal fundus images. This system combines **EfficientNet** for feature extraction and **Graph Convolutional Networks (GCNs)** for relational learning on image features, leveraging spatial dependencies between samples for improved diagnostic accuracy.

---

## 📌 Features

- EfficientNet-based deep feature extraction from fundus images.
- Graph construction using extracted features to capture similarity.
- GCN-based disease classification.
- Graph coarsening and refinement for optimized learning.
- Comparative performance analysis between base GCN and refined GCN.

---

## 🗂️ Dataset

"https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification"

The model was trained and evaluated on a curated dataset containing labeled retinal images across 4 categories:

- **Cataract**
- **Glaucoma**
- **Diabetic Retinopathy**
- **Normal**

## 🏗️ Project Structure

<pre> <code> ``` Eye-Disease-Detection-GCN/ ├── data/ │ ├── raw/ # Original dataset (4 disease classes) │ │ ├── diabetic_retinopathy/ │ │ ├── cataract/ │ │ ├── glaucoma/ │ │ └── normal/ │ ├── preprocessed/ # Preprocessed images │ │ ├── diabetic_retinopathy/ │ │ ├── cataract/ │ │ ├── glaucoma/ │ │ └── normal/ │ └── graph/ # Graph data (EfficientNet features, labels, edges) │ ├── features.npy │ ├── labels.npy │ └── edge_index.npy ├── models/ │ ├── efficientnet_features.py # Extract features │ └── gcn_model.py # GCN architecture ├── graph/ │ ├── build_graph.py # Constructs edge_index │ ├── graph_coarsening.py # Applies coarsening │ └── graph_refinement.py # Refines the graph ├── train/ │ ├── train_gcn.py # Trains base GCN │ ├── train_refined_gcn.py # Trains refined GCN │ └── compare_accuracy.py # Compares model accuracies ├── outputs/ # Models and result files ├── README.md └── requirements.txt ``` </code> </pre>
