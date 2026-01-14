---
title: "Bài 1: Giới thiệu về MLP (Multi-Layer Perceptron)"
date: 2026-01-14
categories:
  - Machine Learning
tags:
  - MLP
  - Deep Learning
toc: true
---

## 1. Mở đầu
Chào mừng đến với blog của Phát. Đây là bài viết đầu tiên về mạng nơ-ron.

## 2. Công thức toán học
Ví dụ về công thức hàm Loss trong Latex:

$$L = -\frac{1}{N} \sum_{i=1}^{N} (y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i))$$

## 3. Code ví dụ
Thử hiển thị code Python:

```python
import torch
import torch.nn as nn

# Định nghĩa một mạng MLP đơn giản
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
print(model)
