# Reproducing Dropout on MNIST

**A reproduction report on *Dropout: A Simple Way to Prevent Neural Networks from
Overfitting* (Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov, JMLR 2014)**

Author: Dantu Bhavyasree

---

## Abstract

Standard neural networks with many parameters can memorize small training datasets
instead of learning general patterns. This is called overfitting. Using many
different networks and averaging their predictions can reduce this problem, but
it is expensive. Dropout (Srivastava et al., 2014) approximates this averaging
inside a *single* network by randomly dropping hidden units during training. I
reproduced the MNIST feed-forward experiment from the paper using a fully
connected net with three hidden layers of 1024 ReLU units
(`784 → 1024 → 1024 → 1024 → 10`), trained once **without dropout** and once
**with dropout** (input `p = 0.2`, hidden `p = 0.5`), differing in nothing else.
The no-dropout net reaches a test error of **3.22%** (96.78% accuracy) and
memorizes the training set; the dropout net reaches **2.70%** (97.30% accuracy)
with a train–validation gap of only 0.05 points instead of 2.88. Dropout
therefore lowered my test error by 0.52 points (≈16% relative) in the same
direction as the paper, though my absolute errors are higher than the paper's
1.25% because I train for 3,140 weight updates instead of roughly a million. The
experiment, the two training curves and the final test accuracies are produced by
a single notebook, `notebooks/MNIST_Dropout_Experiment.ipynb`.

## 1. Introduction

Deep neural networks can perform poorly on new data when they have too many
parameters compared to the amount of training data. Srivastava et al. (2014)
introduce **dropout**: during training, each hidden unit is temporarily removed
with some probability, so no unit can rely on the presence of any particular
other unit. This prevents *co-adaptation* of feature detectors and forces the
network to build more robust, redundant representations.

At test time, dropout is *not* applied. Instead the full network is used, and
because the dropped units are rescaled by `1/(1-p)` during training
("inverted dropout"), the test forward pass needs no extra correction and
approximately averages an exponential number of "thinned" networks.

On MNIST, with a 3-layer network of 1024 ReLU units per layer, the paper reports
a test error of **1.25%**, down from the ≈1.60% of a standard full net. In this
report I reproduce the *basic architecture* version of that experiment
end-to-end and compare my numbers with the paper.

## 2. Dropout in brief

For a unit *h* retained with probability *p* during training:

```
  drop(h) = 0           with probability 1 - p         (training)
            h / (1-p)   with probability p            (training)
            h           always                        (test, "inverted dropout")
```

This is Eq. (5) of the paper, and it is exactly what `nn.Dropout` implements in
PyTorch. Because dropout thins the network during training, the training error
of a dropout net is *higher* at a given epoch than the same net without
dropout — dropout nets simply train slower — but they keep generalizing, so
their *test* error eventually overtakes the no-dropout net.

## 3. Experimental setup

| Component | Value |
|---|---|
| Dataset | MNIST: 60,000 train images, 28×28 grayscale |
| Split | 20,000 train + 5,000 validation + 5,000 test (seed 42; the remaining 30,000 images are left unused) |
| Architecture | `784 → 1024 (ReLU) → 1024 (ReLU) → 1024 (ReLU) → 10` |
| Optimizer | SGD, lr = 0.01, momentum = 0.95 |
| Minibatch | 128 (157 batches per epoch) |
| Epochs | 20 → 3,140 weight updates |
| Dropout net | input retention `p = 0.8` (dropout 0.2); hidden retention `p = 0.5` |
| Loss | softmax cross-entropy |
| Implementation | PyTorch; single notebook in `notebooks/`, run on Google Colab |

Two models are trained identically except for the presence of dropout: a
**no-dropout** net and a **dropout** net (dropout only active during training;
accuracy is always measured in eval mode so the comparison is fair). The
reported test accuracy is measured once, after epoch 20, on the 5,000-image test
split that was held out of the training pool and never seen during training.

## 4. Results

### 4.1 My results

| Model | Train acc. (epoch 20) | Validation acc. (epoch 20) | Test acc. | Test error |
|---|---|---|---|---|
| No dropout | 100.00% | 97.12% | 96.78% | 3.22% |
| Dropout (input 0.2, hidden 0.5) | 97.37% | 97.32% | 97.30% | **2.70%** |

Headline numbers from my run:

- **Test error 3.22% → 2.70%** with dropout: −0.52 points, a ≈16% relative
  reduction (26 of the 5,000 test images).
- **Train–validation gap 2.88 → 0.05 points.** The no-dropout net drives training
  accuracy to 100% while validation stalls at 97.12%; the dropout net ends with
  train 97.37% and validation 97.32%, i.e. it stops memorizing.
- **Dropout learns more slowly.** At epoch 1 the dropout net is at 39.84% train
  accuracy against 55.77% for the no-dropout net, and it only catches up on
  training accuracy around epoch 13. On validation accuracy the two runs are
  neck-and-neck until epoch 15 (both 97.12%), and from epoch 16 onward the
  dropout run is higher in every remaining epoch (97.26 / 97.30 / 97.42 / 97.32 /
  97.32 against 97.04 / 97.02 / 97.08 / 97.14 / 97.12).

> Figures: `results/without_dropout.png` and `results/with_dropout.png` show the
> train and validation accuracy curves for the two runs, written by cells 9 and
> 10 of the notebook.

### 4.2 Comparison with the paper (Table 2, feed-forward MNIST nets)

| Model | Test error | Source |
|---|---|---|
| Standard net, 2 layers, 800 logistic units (Simard et al., 2003) | 1.60% | paper |
| **My no-dropout net, 3 × 1024 ReLU, 3,140 updates** | **3.22%** | this reproduction |
| **My dropout net, 3 × 1024 ReLU, input 0.2 / hidden 0.5, 3,140 updates** | **2.70%** | this reproduction |
| Dropout net, 3 layers, 1024 logistic units | 1.35% | paper |
| Dropout net, 3 layers, 1024 ReLU units | 1.25% | paper |
| Dropout net + max-norm, 3 layers, 1024 ReLU units | 1.06% | paper |
| Dropout net + max-norm, 2 layers, 8192 ReLU units | 0.95% | paper |

My dropout net is **1.45 points worse** than the paper's comparable 3 × 1024 ReLU
dropout net (2.70% vs 1.25%), and my no-dropout net is **1.62 points worse** than
the standard net the paper compares against (3.22% vs 1.60%). In other words
both of my runs are weaker in absolute terms, but the *ordering and the sign of
the effect* match the paper: adding dropout to the same architecture reduces the
test error.

### 4.3 Observations

1. **Dropout helps, but less than in the paper.** The paper's headline gap is
   1.60% → 1.25% (−0.35 points); mine is 3.22% → 2.70% (−0.52 points). The
   relative improvement is of the same order, from a weaker starting point.
2. **Overfitting in the no-dropout net.** Training accuracy saturates at 100%
   from epoch 16 onwards while validation accuracy oscillates around 97.1% — the
   signature of memorization with no generalization gain.
3. **Dropout trains slower, as the paper predicts.** It trails on training
   accuracy for the first twelve epochs and only overtakes on validation accuracy
   late in training. This is the classic dropout learning-curve signature (slow
   train, still-improving validation), and it is why short runs understate
   dropout's benefit.
4. **Single run, no global seed.** Only the data split is seeded (42); weight
   initialization and batch order are not, so repeated runs will differ. A 0.52
   point gap on 5,000 test images is 26 images, and the standard error of a
   single accuracy estimate at this level is ≈0.24 points, so this gap is
   suggestive rather than statistically strong. Repeating both runs over several
   seeds would be needed to state it as a firm result.
5. **Test split is small and comes from the training pool.** I hold out 5,000 of
   the 60,000 training images instead of using the official 10,000-image test
   set, which adds noise to every number in section 4.1.

## 5. Comparison with the Original Paper

The paper reports a 1.25% MNIST test error using a larger network, roughly a
million weight updates, input-layer dropout and max-norm constraints. My
experiment uses 3,140 weight updates (20 epochs over 20,000 images), a learning
rate of 0.01 with momentum 0.95, no max-norm constraint, no data augmentation,
and discards 30,000 of the 60,000 available images. Section 4.3 explains why
this matters most for the dropout net: dropout's advantage grows with training
length, and at 3,140 updates the dropout run has barely moved past the no-dropout
run. Matching the paper more closely would mean training for far more epochs at a
higher learning rate, adding max-norm constraints, and evaluating on the official
10,000-image test set.

## 6. Conclusion

I set up a controlled MNIST experiment in which the only difference between the
two runs is the presence of dropout. The no-dropout network memorized the
training set (100% train accuracy) for a 3.22% test error, while dropout
prevented hidden units from depending on each other too strongly: training was
slower, the train–validation gap shrank from 2.88 to 0.05 points, and the test
error fell to 2.70%. This reproduces the qualitative claim of the 2014 paper —
dropout trades slower training for less overfitting and better test error — at
a much shorter training budget than the paper used, and with a gap that a single
seeded run cannot separate from run-to-run noise.

## References

- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R.
  (2014). *Dropout: A simple way to prevent neural networks from overfitting.*
  Journal of Machine Learning Research, 15, 1929–1958.
- Simard, P. Y., Steinkraus, D., & Platt, J. C. (2003). *Best practices for
  convolutional neural networks applied to visual document analysis.* ICDAR.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning
  applied to document recognition.* Proceedings of the IEEE.

---
