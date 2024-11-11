# SynergiProtoNet_Bangla_OCR_Few_Shot_Learning
This project implements the SynergiProtoNet, a few-shot learning model for recognizing handwritten characters and digits in Bangla script. Based on the methodologies described in our paper, this model demonstrates the ability to perform high-accuracy recognition with limited labeled data, addressing challenges inherent to low-resource languages. It also demonstrates comparative analysis among different classical few shot learning approaches.

## Prerequisites

Before you run this notebook, ensure that you have the following:

- Python 3.10 or later
- Pip package manager
- Access to a GPU (recommended for faster computation)

## Performance Analysis
# Monolingual Intra-Dataset Evaluation

| Network          | 1-shot Acc(%) | 1-shot F1 | 1-shot Recall | 1-shot Precision | 5-shot Acc(%) | 5-shot F1 | 5-shot Recall | 5-shot Precision | 10-shot Acc(%) | 10-shot F1 | 10-shot Recall | 10-shot Precision |
|------------------|---------------|-----------|---------------|------------------|---------------|-----------|---------------|------------------|----------------|------------|----------------|-------------------|
| Matching [19]    | 69.64         | 0.69      | Recall1       | Precision1       | 38.66         | 0.39      | Recall5       | Precision5       | 36.36          | 0.36       | Recall10       | Precision10       |
| Simpleshot [22]  | 69.42         | 0.70      | Recall1       | Precision1       | 75.68         | 0.78      | Recall5       | Precision5       | 82.9           | 0.82       | Recall10       | Precision10       |
| Relation [20]    | 77.10         | 0.76      | Recall1       | Precision1       | 85.58         | 0.87      | Recall5       | Precision5       | 83.2           | 0.83       | Recall10       | Precision10       |
| BD-CSPN [21]     | 69.68         | 0.71      | Recall1       | Precision1       | 76.58         | 0.76      | Recall5       | Precision5       | 83.48          | 0.82       | Recall10       | Precision10       |
| Prototypical [23]| 74.48         | 0.74      | Recall1       | Precision1       | 87.88         | 0.87      | Recall5       | Precision5       | 87.56          | 0.87       | Recall10       | Precision10       |
| SynergiProtoNet  | **79.1**      | **0.79**  | Recall1       | Precision1       | **88.95**     | **0.88**  | Recall5       | Precision5       | **90.04**      | **0.90**   | Recall10       | Precision10       |
