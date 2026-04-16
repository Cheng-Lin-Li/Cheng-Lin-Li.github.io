---
layout: post
title: Tips for a Neural network model training
tags: training
---

<!-- more -->

---
Table of content:

{: class="table-of-content"}
* TOC
{:toc}

---
# Tips for a Neural network model training
## The Major 3 Steps in a Neural network Model Training

- **Step 1:** What is the task?  
- **Step 2:** What is the best function for the task? 
- **Step 3:** Choose the best functional scope.

## Table of methods/functions for their benefit:

| Method                                 | Which is the step this method apply | Benefits |                                                            |
|----------------------------------------|-------------------------------------|---------------------------------------------------------------------|
| Adagrad, RMSProp, Momentum, Adam, etc. | Find the best function | Better Optimization |
| AdamW                                  | Find the best function| Better Generalization (cf. Adam), Better Optimization (cf. Vanilla gradient Descent) |
| Dropout                                | Find the best function| Better Generalization|
| Weight Decay                           | Find the best function| Better Generalization|
| Initialization (e.g., pre-train)       | Find the best function| Better Optimization, Better Generalization|
| CNN (e.g., for image)                  | Change search the scope of the function| Better Generalization|
| Skip Connection                        | Change search the scope of the function| Better Optimization |
| Normalization                          | Change search the scope of the function| Better Optimization, (Sometimes Better Generalization)|
| Do not use accuracy as loss            | What I am looking for | Better Optimization  |
| More training data                     | What I am looking for | Better Generalization|
| Data Augmentation (e.g. Mixup)         | What I am looking for | Better Generalization|
| Semi-supervised (e.g., Entropy, Graph) | What I am looking for | Better Generalization|
| Parameter Regularization               | What I am looking for | Better Generalization|


# 訓練類神經網路的各種訣竅
### 類神經網路訓練的三個主要步驟

- **Step 1:** 我要找什麼?  
- **Step 2:** 我有哪些函式可以選擇? 
- **Step 3:** 選一個最好的函式範圍.

### 各種方法的列表與其好處:

| 方法名 | 改了那一個步驟 | 帶來什麼好處 |
|---|---|---|
|Adagrad, RMSProp, Momentum, Adam, etc.| 找最好的函式 | Better Optimization |
|AdamW | 找最好的函式 | Better Generalization (cf. Adam), Better Optimization (cf. Vanilla)|
|Dropout | 找最好的函式 | Better Generalization |
|Weight Decay | 找最好的函式 | Better Generalization|
|Initialization (e.g., pre-train)| 找最好的函式 | Better Optimization, Better Generalization|
|CNN (e.g., for image)| 改變函式搜尋範圍 | Better Generalization|
|Skip Connection | 改變函式搜尋範圍 | Better Optimization|
|Normalization | 改變函式搜尋範圍 | Better Optimization, (Sometimes Better Generalization)|
|Do not use accuracy as loss | 我要找什麼 | Better Optimization |
|More training data | 我要找什麼 | Better Generalization|
|Data Augmentation (e.g. Mixup)| 我要找什麼 | Better Generalization|
|Semi-supervised (e.g., Entropy, Graph)| 我要找什麼 | Better Generalization|
|Parameter Regularization| 我要找甚麼 | Better Generalization|


Reference:
https://speech.ee.ntu.edu.tw/~hylee/GenAI-ML/2025-fall-course-data/TrainingTip.pdf

