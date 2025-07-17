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
<pre> <code> ``` <PASTE Eye-Disease-Detection-GCN
│   README.md
│   requirements.txt
│
├───data
│   ├───raw
│   │   ├───cataract
│   │   ├───glaucoma
│   │   ├───normal
│   │   └───diabetic_retinopathy
│   └───preprocessed
│       ├───...
├───models
│       gcn_model.py
│       efficientnet_features.py
├───graph
│       graph_coarsening.py
│       graph_refinement.py
│       build_graph.py
├───train
│       train_gcn.py
│       train_refined_gcn.py
│       compare_accuracy.py
└───outputs
> ``` </code> </pre>
