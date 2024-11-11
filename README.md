# SynergiProtoNet_Bangla_OCR_Few_Shot_Learning
This project implements the SynergiProtoNet, a few-shot learning model for recognizing handwritten characters and digits in Bangla script. Based on the methodologies described in our paper, this model demonstrates the ability to perform high-accuracy recognition with limited labeled data, addressing challenges inherent to low-resource languages. It also demonstrates comparative analysis among different classical few shot learning approaches.

## Prerequisites

Before you run this notebook, ensure that you have the following:

- Python 3.10 or later
- Pip package manager
- Git CLI 
- Access to a GPU (recommended for faster computation)

# Performance Analysis
## Monolingual Intra-Dataset Evaluation

| Network          | 1-shot Acc(%) | 1-shot F1 | 1-shot Recall | 1-shot Precision | 5-shot Acc(%) | 5-shot F1 | 5-shot Recall | 5-shot Precision | 10-shot Acc(%) | 10-shot F1 | 10-shot Recall | 10-shot Precision |
|------------------|---------------|-----------|---------------|------------------|---------------|-----------|---------------|------------------|----------------|------------|----------------|-------------------|
| Matching   | 69.64         | 0.69      |  0.69     |  0.69        | 38.66         | 0.39      | 0.39       |  0.39       | 36.36          | 0.36       | 0.36      | 0.36      |
| Simpleshot   | 69.42         | 0.70      | 0.71       | 0.70       | 75.68         | 0.78      | 0.78       | 0.78       | 82.9           | 0.82       | Recall10       | Precision10       |
| Relation [20]    | 77.10         | 0.76      | Recall1       | Precision1       | 85.58         | 0.87      | Recall5       | Precision5       | 83.2           | 0.83       | Recall10       | Precision10       |
| BD-CSPN [21]     | 69.68         | 0.71      | Recall1       | Precision1       | 76.58         | 0.76      | Recall5       | Precision5       | 83.48          | 0.82       | Recall10       | Precision10       |
| Prototypical [23]| 74.48         | 0.74      | Recall1       | Precision1       | 87.88         | 0.87      | Recall5       | Precision5       | 87.56          | 0.87       | Recall10       | Precision10       |
| SynergiProtoNet  | **79.1**      | **0.79**  | Recall1       | Precision1       | **88.95**     | **0.88**  | Recall5       | Precision5       | **90.04**      | **0.90**   | Recall10       | Precision10       |








## Monolingual Inter-Dataset Evaluation

| Network          | 1-shot Acc(%) | 1-shot F1 | 1-shot Recall | 1-shot Precision | 5-shot Acc(%) | 5-shot F1 | 5-shot Recall | 5-shot Precision | 10-shot Acc(%) | 10-shot F1 | 10-shot Recall | 10-shot Precision |
|------------------|---------------|-----------|---------------|------------------|---------------|-----------|---------------|------------------|----------------|------------|----------------|-------------------|
| Matching [19]    | 48.6          | 0.52      | Recall1       | Precision1       | 41.8          | 0.39      | Recall5       | Precision5       | 26.9           | 0.26       | Recall10       | Precision10       |
| Simpleshot [22]  | 51.34         | 0.55      | Recall1       | Precision1       | 63.54         | 0.63      | Recall5       | Precision5       | 64.94          | 0.66       | Recall10       | Precision10       |
| Relation [20]    | 56.46         | 0.58      | Recall1       | Precision1       | 74.02         | 0.72      | Recall5       | Precision5       | 65.04          | 0.67       | Recall10       | Precision10       |
| BD-CSPN [21]     | 50.84         | 0.5       | Recall1       | Precision1       | 60.3          | 0.6       | Recall5       | Precision5       | 69.2           | 0.7        | Recall10       | Precision10       |
| Prototypical [23]| 54.28         | 0.55      | Recall1       | Precision1       | 76.18         | 0.75      | Recall5       | Precision5       | 76.24          | 0.75       | Recall10       | Precision10       |
| SynergiProtoNet  | **59.02**     | **0.59**  | Recall1       | Precision1       | **77.68**     | **0.78**  | Recall5       | Precision5       | **81.36**      | **0.83**   | Recall10       | Precision10       |




## Cross-Lingual Performance Analysis

| Network          | 1-shot Acc(%) | 1-shot F1 | 1-shot Recall | 1-shot Precision | 5-shot Acc(%) | 5-shot F1 | 5-shot Recall | 5-shot Precision | 10-shot Acc(%) | 10-shot F1 | 10-shot Recall | 10-shot Precision |
|------------------|---------------|-----------|---------------|------------------|---------------|-----------|---------------|------------------|----------------|------------|----------------|-------------------|
| Matching [19]    | 52.84         | 0.52      | Recall1       | Precision1       | 26.68         | 0.27      | Recall5       | Precision5       | 37.58          | 0.38       | Recall10       | Precision10       |
| Simpleshot [22]  | 42.92         | 0.46      | Recall1       | Precision1       | 56.14         | 0.53      | Recall5       | Precision5       | 55.10          | 0.54       | Recall10       | Precision10       |
| BD-CSPN [21]     | 44.58         | 0.45      | Recall1       | Precision1       | 54.72         | 0.52      | Recall5       | Precision5       | 57.72          | 0.57       | Recall10       | Precision10       |
| Prototypical [23]| 53.64         | 0.55      | Recall1       | Precision1       | 76.74         | 0.77      | Recall5       | Precision5       | 79.48          | 0.79       | Recall10       | Precision10       |
| Relation [20]    | 61.12         | 0.61      | Recall1       | Precision1       | 74.02         | 0.72      | Recall5       | Precision5       | 81.93          | 0.82       | Recall10       | Precision10       |
| SynergiProtoNet  | 58.59         | 0.55      | Recall1       | Precision1       | 76.84         | 0.77      | Recall5       | Precision5       | 82.12          | 0.82       | Recall10       | Precision10       |




## Split Digit Testing

| Network          | 1-shot Acc(%) | 1-shot F1 | 1-shot Recall | 1-shot Precision | 5-shot Acc(%) | 5-shot F1 | 5-shot Recall | 5-shot Precision | 10-shot Acc(%) | 10-shot F1 | 10-shot Recall | 10-shot Precision |
|------------------|---------------|-----------|---------------|------------------|---------------|-----------|---------------|------------------|----------------|------------|----------------|-------------------|
| Matching [19]    | 73.67         | 0.74      | Recall1       | Precision1       | 33.1          | 0.29      | Recall5       | Precision5       | 32.17          | 0.33       | Recall10       | Precision10       |
| BD-CSPN [21]     | 55.43         | 0.59      | Recall1       | Precision1       | 64.43         | 0.64      | Recall5       | Precision5       | 69.17          | 0.68       | Recall10       | Precision10       |
| Simpleshot [22]  | 65.6          | 0.61      | Recall1       | Precision1       | 72.9          | 0.7       | Recall5       | Precision5       | 76.97          | 0.76       | Recall10       | Precision10       |
| Relation [20]    | 64.97         | 0.63      | Recall1       | Precision1       | 79.83         | 0.8       | Recall5       | Precision5       | 81.37          | 0.83       | Recall10       | Precision10       |
| Prototypical [23]| 66.57         | 0.65      | Recall1       | Precision1       | 76.33         | 0.75      | Recall5       | Precision5       | 87.1           | 0.86       | Recall10       | Precision10       |
| SynergiProtoNet  | 37.4          | 0.37      | Recall1       | Precision1       | 83.73         | 0.83      | Recall5       | Precision5       | 88.3           | 0.87       | Recall10       | Precision10       |


