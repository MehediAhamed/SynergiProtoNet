# SynergiProtoNet_Bangla_OCR_Few_Shot_Learning
This project implements the SynergiProtoNet, a few-shot learning model for recognizing handwritten characters and digits in Bangla script. Based on the methodologies described in our paper, this model demonstrates the ability to perform high-accuracy recognition with limited labeled data, addressing challenges inherent to low-resource languages. It also demonstrates comparative analysis among different classical few shot learning approaches.

# Prerequisites

Before you run this notebook, ensure that you have the following:

- Python 3.10 or later
- Pip package manager
- Git CLI 
- Access to a GPU (recommended for faster computation)

# Download Datasets 
- CMATERdb- https://www.kaggle.com/datasets/mostafiz53/basicfinal
- NumtaDB- https://www.kaggle.com/datasets/BengaliAI/numta
- BanglaLekha-Isolated- https://www.sciencedirect.com/science/article/pii/S2352340917301117
- Devanagari- https://www.kaggle.com/datasets/suvooo/hindi-character-recognition

# Performance Analysis
## Monolingual Intra-Dataset Evaluation


| Network          | 1-shot Acc(%) | 1-shot F1 | 1-shot Recall | 1-shot Precision | 5-shot Acc(%) | 5-shot F1 | 5-shot Recall | 5-shot Precision | 10-shot Acc(%) | 10-shot F1 | 10-shot Recall | 10-shot Precision |
|------------------|---------------|-----------|---------------|------------------|---------------|-----------|---------------|------------------|----------------|------------|----------------|-------------------|
| Matching     | 69.64         | 0.69      | 0.69          | 0.69             | 38.66         | 0.39      | 0.39          | 0.39             | 36.36          | 0.36       | 0.36           | 0.36              |
| Simpleshot   | 69.42         | 0.70      | 0.71          | 0.70             | 75.68         | 0.78      | 0.78          | 0.77             | 82.9           | 0.82       | 0.82           | 0.82              |
| Relation     | 77.10         | 0.76      | 0.76          | 0.76             | 85.58         | 0.87      | 0.87          | 0.87             | 83.2           | 0.83       | 0.83           | 0.83              |
| BD-CSPN      | 69.68         | 0.71      | 0.71          | 0.71             | 76.58         | 0.76      | 0.76          | 0.76             | 83.48          | 0.82       | 0.82           | 0.82              |
| Prototypical | 74.48         | 0.74      | 0.75          | 0.74             | 87.88         | 0.87      | 0.87          | 0.87             | 87.56          | 0.87       | 0.87           | 0.87              |
| SynergiProtoNet  | 79.1          | 0.79      | 0.79          | 0.79             | 88.95         | 0.88      | 0.88          | 0.88             | 90.04          | 0.90       | 0.90           | 0.90              |







## Monolingual Inter-Dataset Evaluation

| Network            | 1-shot Acc(%) | 1-shot F1 | 1-shot Recall | 1-shot Precision | 5-shot Acc(%) | 5-shot F1 | 5-shot Recall | 5-shot Precision | 10-shot Acc(%) | 10-shot F1 | 10-shot Recall | 10-shot Precision |
|--------------------|---------------|-----------|---------------|------------------|---------------|-----------|---------------|------------------|----------------|------------|----------------|-------------------|
| Matching       | 48.6          | 0.52      | 0.51          | 0.52             | 41.8          | 0.39      | 0.4           | 0.39             | 26.9           | 0.26       | 0.26           | 0.26              |
| Simpleshot     | 51.34         | 0.55      | 0.54          | 0.54             | 63.54         | 0.63      | 0.63          | 0.63             | 64.94          | 0.66       | 0.66           | 0.66              |
| BD-CSPN        | 50.84         | 0.5       | 0.5           | 0.51             | 60.3          | 0.6       | 0.6           | 0.6              | 69.2           | 0.7        | 0.7            | 0.7               |
| Prototypical   | 54.28         | 0.55      | 0.56          | 0.55             | 76.18         | 0.75      | 0.75          | 0.75             | 76.24          | 0.75       | 0.75           | 0.75              |
| Synergi w/o pretrain| 44.5         | 0.45      | 0.44          | 0.44             | 65.16         | 0.66      | 0.66          | 0.66             | 68.76          | 0.68       | 0.68           | 0.68              |
| SynergiProtoNet    | 59.02         | 0.59      | 0.59          | 0.59             | 77.68         | 0.78      | 0.78          | 0.78             | 81.36          | 0.83       | 0.83           | 0.83              |
| Relation      | 56.46         | 0.58      | 0.58          | 0.58             | 74.02         | 0.72      | 0.72          | 0.72             | 65.04          | 0.67       | 0.65           | 0.65              |


## Cross-Lingual Performance Analysis


| Network            | 1-shot Acc(%) | 1-shot F1 | 1-shot Recall | 1-shot Precision | 5-shot Acc(%) | 5-shot F1 | 5-shot Recall | 5-shot Precision | 10-shot Acc(%) | 10-shot F1 | 10-shot Recall | 10-shot Precision |
|--------------------|---------------|-----------|---------------|------------------|---------------|-----------|---------------|------------------|----------------|------------|----------------|-------------------|
| Matching      | 52.84         | 0.52      | 0.52          | 0.52             | 26.68         | 0.27      | 0.27          | 0.27             | 37.58          | 0.38       | 0.38           | 0.38              |
| Simpleshot     | 42.92         | 0.46      | 0.46          | 0.46             | 56.14         | 0.53      | 0.53          | 0.53             | 55.10          | 0.54       | 0.54           | 0.54              |
| BD-CSPN       | 44.58         | 0.45      | 0.45          | 0.45             | 54.72         | 0.52      | 0.52          | 0.53             | 57.72          | 0.57       | 0.57           | 0.57              |
| Prototypical  | 53.64         | 0.55      | 0.56          | 0.55             | 76.74         | 0.77      | 0.77          | 0.77             | 79.48          | 0.79       | 0.79           | 0.79              |
| Synergi w/o pretrain | 50.46       | 0.51      | 0.51          | 0.51             | 71.14         | 0.70      | 0.70          | 0.70             | 76.98          | 0.76       | 0.76           | 0.76              |
| SynergiProtoNet    | 58.59         | 0.55      | 0.55          | 0.55             | 76.84         | 0.77      | 0.77          | 0.77             | 82.12          | 0.82       | 0.82           | 0.82              |
| Relation      | 61.12         | 0.61      | 0.61          | 0.61             | 74.02         | 0.72      | 0.72          | 0.72             | 81.93          | 0.82       | 0.82           | 0.82              |





## Split Digit Testing

| Network              | 1-shot Acc(%) | 1-shot F1 | 1-shot Recall | 1-shot Precision | 5-shot Acc(%) | 5-shot F1 | 5-shot Recall | 5-shot Precision | 10-shot Acc(%) | 10-shot F1 | 10-shot Recall | 10-shot Precision |
|----------------------|---------------|-----------|---------------|------------------|---------------|-----------|---------------|------------------|----------------|------------|----------------|-------------------|
| Matching       | 73.67         | 0.74      | 0.75          | 0.75             | 33.1          | 0.29      | 0.34          | 0.34             | 32.17          | 0.33       | 0.34           | 0.34              |
| BD-CSPN         | 55.43         | 0.59      | 0.59          | 0.59             | 64.43         | 0.64      | 0.64          | 0.64             | 69.17          | 0.68       | 0.68           | 0.68              |
| Simpleshot      | 65.6          | 0.61      | 0.62          | 0.61             | 72.9          | 0.7       | 0.71          | 0.73             | 76.97          | 0.76       | 0.76           | 0.76              |
| Relation        | 64.97         | 0.63      | 0.64          | 0.64             | 79.83         | 0.8       | 0.79          | 0.8              | 81.37          | 0.83       | 0.83           | 0.83              |
| Prototypical     | 66.57         | 0.65      | 0.65          | 0.65             | 76.33         | 0.75      | 0.74          | 0.74             | 87.1           | 0.86       | 0.86           | 0.86              |
| Synergi w/o pretrain | 46.47         | 0.48      | 0.47          | 0.47             | 52.6          | 0.53      | 0.53          | 0.53             | 57.6           | 0.57       | 0.57           | 0.57              |
| SynergiProtoNet      | 37.4          | 0.37      | 0.37          | 0.37             | 83.73         | 0.83      | 0.83          | 0.83             | 88.3           | 0.87       | 0.87           | 0.87              |







# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Cite
@INPROCEEDINGS{10951048,
  author={Ahamed, Mehedi and Kabir, Radib Bin and Dipto, Tawsif Tashwar and Al Mushabbir, Mueeze and Ahmed, Sabbir and Kabir, Md. Hasanul},
  booktitle={2024 6th International Conference on Sustainable Technologies for Industry 5.0 (STI)}, 
  title={Performance Analysis of Few-Shot Learning Approaches for Bangla Handwritten Character and Digit Recognition}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/STI64222.2024.10951048},
  ISSN={2996-170X},
  month={Dec},}


