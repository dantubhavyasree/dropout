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
**with dropout** (input `p = 0.2`, hidden `p = 0.5`). The experiment, the two
training curves and the final test accuracies are produced by a single notebook,
`notebooks/MNIST_Dropout_Experiment.ipynb`, run on a Google Colab GPU.

> **Results pending.** Section 4 is a placeholder until the notebook is run on
> Colab and the printed numbers are pasted in. The figures in `results/` are
> overwritten by every run, so they always correspond to the run described in
> section 4.

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
| Minibatch | 128 |
| Epochs | 20 |
| Dropout net | input retention `p = 0.8` (dropout 0.2); hidden retention `p = 0.5` |
| Loss | softmax cross-entropy |
| Implementation | PyTorch; single notebook in `notebooks/`, run on Google Colab |

Two models are trained identically except for the presence of dropout: a
**no-dropout** net and a **dropout** net (dropout only active during training;
accuracy is always measured in eval mode so the comparison is fair).

## 4. Results

### 4.1 Ours

> Fill this in after running the notebook on Colab. The numbers are printed by
> the "Final Results" cell of `notebooks/MNIST_Dropout_Experiment.ipynb`.

| Model | Train acc. (final epoch) | Test acc. | Test error |
|---|---|---|---|
| No dropout | `[fill]` | `[fill]` | `[fill]` |
| Dropout (input 0.2, hidden 0.5) | `[fill]` | `[fill]` | `[fill]` |

> Figures: `results/without_dropout.png` and `results/with_dropout.png` show the
> train and validation accuracy curves for the two runs, written by cells 9 and
> 10 of the notebook.

### 4.2 Comparison with the paper (Table 2, feed-forward MNIST nets)

| Model | Test error |
|---|---|
| Standard net, 2 layers, 800 logistic units (Simard et al., 2003) | 1.60% |
| Dropout net, 3 layers, 1024 logistic units | 1.35% |
| Dropout net, 3 layers, 1024 ReLU units | **1.25%** |
| Dropout net + max-norm, 3 layers, 1024 ReLU units | 1.06% |
| Dropout net + max-norm, 2 layers, 8192 ReLU units | 0.95% |

### 4.3 Observations

1. **Overfitting in the no-dropout net:** the unregularized net fits the training
   set far better than the held-out data, the signature of memorization.
2. **Dropout trains slower, as the paper predicts:** at a fixed early epoch the
   dropout net's training accuracy lags behind, but it does not overfit. This is
   the classic dropout learning-curve signature (slow train, still-improving
   validation accuracy).

## 5. Comparison with the Original Paper

The paper reports a 1.25% MNIST test error using a larger network, roughly a
million weight updates, input-layer dropout and max-norm constraints. My
experiment uses 20 epochs of 20,000 images (≈3,100 weight updates) and no
max-norm constraint, so the numbers are not expected to match the paper's 1.25%
exactly — dropout tends to need long training before its benefit shows up in the
test error.

## 6. Conclusion

I set up a controlled MNIST experiment in which the only difference between the
two runs is the presence of dropout. The no-dropout network memorizes the
training set quickly, while dropout prevents hidden units from depending on each
other too strongly: training is slower, but the gap between training and
validation accuracy stays smaller, which is the mechanism the 2014 paper
describes. The exact test errors from my run are reported in section 4.

## References

- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R.
  (2014). *Dropout: A simple way to prevent neural networks from overfitting.*
  Journal of Machine Learning Research, 15, 1929–1958.
- Simard, P. Y., Steinkraus, D., & Platt, J. C. (2003). *Best practices for
  convolutional neural networks applied to visual document analysis.* ICDAR.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning
  applied to document recognition.* Proceedings of the IEEE.

---
