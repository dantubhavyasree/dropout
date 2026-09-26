  # Reproducing Dropout on MNIST

**A reproduction report on *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (Srivastava et al., 2014)**

**Author:** Dantu Bhavyasree

---

## Abstract

Neural networks can sometimes learn the training data too closely. This is called **overfitting**. Dropout is a technique that helps reduce overfitting by randomly turning off some neurons during training.

In this project, I reproduced the basic MNIST feed-forward neural network experiment from the paper *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. I used a fully connected neural network with three hidden layers of 1024 ReLU neurons. The same network was trained twice: once without dropout and once with dropout.

The network without dropout achieved **96.78% test accuracy**, while the network with dropout achieved **97.30% test accuracy**. The test error decreased from **3.22% to 2.70%** when dropout was used.

The results also show less overfitting with dropout. The no-dropout model reached **100% training accuracy** but **97.12% validation accuracy**, while the dropout model reached **97.37% training accuracy** and **97.32% validation accuracy**.

These results support the main idea of the original paper that dropout can reduce overfitting and improve performance on unseen data.

---

## 1. Introduction

Neural networks learn patterns from training data. However, a network can sometimes learn the training examples too closely instead of learning patterns that work well on new data. This problem is called **overfitting**.

Dropout was introduced by Srivastava et al. (2014) as a way to reduce overfitting.

During training, dropout randomly turns off some neurons. This prevents the network from depending too much on particular neurons and encourages it to learn more general patterns.

Dropout is turned off during testing, so the complete network is used to make predictions.

In this project, I reproduced a basic MNIST experiment from the original paper. I trained the same neural network with and without dropout and compared their training, validation, and test performance.

---

## 2. Dropout in Brief

Dropout randomly turns off some neurons during training.

For example:

<img width="569" height="313" alt="image" src="https://github.com/user-attachments/assets/62379f8d-e957-4e85-9bdd-1eef8fba109f" />


The neurons are temporarily ignored during that training step.

In this experiment:

* **Input dropout:** 20%
* **Hidden-layer dropout:** 50%

Dropout is used only during training. During testing, all neurons are used.

The main purpose of dropout is to reduce overfitting and help the network perform better on data it has not seen before.

---

## 3. Experimental Setup

### 3.1 Dataset

The experiment uses the **MNIST handwritten digit dataset**.

Each image is a 28 × 28 grayscale image containing a handwritten digit from 0 to 9.

For this experiment, the dataset was divided as follows:

| Split      | Number of Images |
| ---------- | ---------------: |
| Training   |           20,000 |
| Validation |            5,000 |
| Testing    |            5,000 |
| Unused     |           30,000 |

The data split used random seed `42`.

The 5,000 test images were selected from the original MNIST training set. The official 10,000-image MNIST test set was not used.

### 3.2 Network Architecture

Both experiments use the same fully connected neural network:

```text
784 → 1024 → 1024 → 1024 → 10
```

Where:

* `784` = 28 × 28 input pixels
* `1024` = first hidden layer
* `1024` = second hidden layer
* `1024` = third hidden layer
* `10` = output classes for digits 0–9

ReLU activation is used in the hidden layers.

### 3.3 Training Settings

| Setting           | Value   |
| ----------------- | ------- |
| Framework         | PyTorch |
| Optimizer         | SGD     |
| Learning Rate     | 0.01    |
| Momentum          | 0.95    |
| Batch Size        | 128     |
| Epochs            | 20      |
| Hidden Activation | ReLU    |
| Input Dropout     | 0.2     |
| Hidden Dropout    | 0.5     |
| Device            | CPU     |
| Training Images   | 20,000  |
| Validation Images | 5,000   |
| Test Images       | 5,000   |

The experiment was run using **Google Colab**.

### 3.4 Two Experiments

Two versions of the same network were trained.

**Experiment 1 — Without Dropout**

```text
784 → 1024 → 1024 → 1024 → 10
```

**Experiment 2 — With Dropout**

```text
784 → Dropout → 1024 → Dropout → 1024 → Dropout → 1024 → 10
```

All other training settings were kept the same.

---

## 4. Results

### 4.1 Final Results

| Model           | Training Accuracy | Validation Accuracy | Test Accuracy | Test Error |
| --------------- | ----------------: | ------------------: | ------------: | ---------: |
| Without Dropout |           100.00% |              97.12% |        96.78% |      3.22% |
| With Dropout    |            97.37% |              97.32% |    **97.30%** |  **2.70%** |

The dropout model achieved:

**97.30% test accuracy**

compared with:

**96.78% test accuracy**

without dropout.

Therefore:

```text
Test accuracy improvement = 97.30% - 96.78%
                           = 0.52 percentage points
```

The test error decreased from:

```text
3.22% → 2.70%
```

---

### 4.2 Training and Validation Accuracy

The no-dropout model reached **100% training accuracy**, while its validation accuracy was **97.12%**.

This gives a difference of:

```text
100.00% - 97.12% = 2.88 percentage points
```

This shows that the model performed much better on the training data than on the validation data.

With dropout:

```text
Training accuracy   = 97.37%
Validation accuracy = 97.32%
```

The difference was only:

```text
97.37% - 97.32% = 0.05 percentage points
```

The much smaller difference suggests that dropout reduced overfitting in this experiment.

---

### 4.3 Learning During Training

The dropout model learned more slowly at the beginning.

After the first epoch:

| Model           | Training Accuracy | Validation Accuracy |
| --------------- | ----------------: | ------------------: |
| Without Dropout |            55.77% |              87.84% |
| With Dropout    |            39.84% |              80.42% |

However, the validation accuracy of the dropout model continued to improve during training.

At epoch 20:

| Model           | Training Accuracy | Validation Accuracy |
| --------------- | ----------------: | ------------------: |
| Without Dropout |           100.00% |              97.12% |
| With Dropout    |            97.37% |              97.32% |

This shows that dropout made training slower but helped reduce the difference between training and validation performance.

---

## 5. Comparison With the Original Paper

The original paper reported a **1.25% test error** for a similar three-layer network with 1024 ReLU units and dropout.

My experiment produced a **2.70% test error**.

| Experiment      | Test Error |
| --------------- | ---------: |
| Original paper  |      1.25% |
| My reproduction |      2.70% |

The results are different because the experimental setup was not exactly the same.

### Main differences

My experiment used:

* 20,000 training images
* 5,000 validation images
* 5,000 test images
* 20 epochs
* 3,140 weight updates
* No max-norm constraint
* No data augmentation
* One training run

The original paper used a larger training setup and trained the network for much longer. It also used additional techniques such as max-norm constraints.

Therefore, this project should be considered a **reproduction of the main experiment and idea**, rather than an exact reproduction of every detail of the original paper.

The important observation is that, under the same setup, the dropout model achieved lower test error than the no-dropout model:

```text
Without Dropout: 3.22%
With Dropout:    2.70%
```

---

## 6. Conclusion

This project compared the same MNIST neural network with and without dropout.

The network without dropout achieved **96.78% test accuracy**, while the network with dropout achieved **97.30% test accuracy**.

Dropout therefore improved test accuracy by **0.52 percentage points** and reduced test error from **3.22% to 2.70%**.

The no-dropout model reached 100% training accuracy but had lower validation accuracy. The dropout model had lower training accuracy but almost the same training and validation accuracy.

This shows that dropout reduced overfitting in this experiment.

Overall, the experiment supports the main idea of the original paper: **dropout can reduce overfitting and improve performance on unseen data.**

The result is not as strong as the result reported in the original paper because this reproduction used fewer training images, fewer training updates, and a different experimental setup.

---

## 7. References

1. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, **"Dropout: A Simple Way to Prevent Neural Networks from Overfitting,"** *Journal of Machine Learning Research*, vol. 15, pp. 1929–1958, 2014.

2. P. Y. Simard, D. Steinkraus, and J. C. Platt, **"Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis,"** ICDAR, 2003.

3. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, **"Gradient-Based Learning Applied to Document Recognition,"** *Proceedings of the IEEE*, 1998.



---
