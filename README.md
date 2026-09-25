# MNIST Dropout Experiment

Reproducing **Dropout: A Simple Way to Prevent Neural Networks from Overfitting**
(Srivastava et al., 2014), focusing on the MNIST feed-forward experiment.

The same network (`784 → 1024 → 1024 → 1024 → 10`, ReLU) is trained twice on
MNIST: **without dropout** and **with dropout** (input `p = 0.2`, hidden
`p = 0.5`). Everything runs in **Google Colab** on a GPU — no local installation
needed.

## Project structure

```
dropout/
│
├── notebooks/
│   └── MNIST_Dropout_Experiment.ipynb   # the whole experiment, start to finish
│
├── results/
│   ├── without_dropout.png              # train/validation accuracy, no dropout
│   └── with_dropout.png                 # train/validation accuracy, with dropout
│
├── PAPER.md                             # full reproduction report
├── README.md
└── requirements.txt
```

## Run it on Colab

1. Open `notebooks/MNIST_Dropout_Experiment.ipynb` in
   [Google Colab](https://colab.research.google.com) (GitHub → *Open in Colab*).
2. *Runtime → Run all*. MNIST downloads itself on the first run.
3. The notebook overwrites `results/without_dropout.png` and
   `results/with_dropout.png` with the curves from your run.
4. Copy those two images back into `results/` and commit them.

## Experimental setup

| Component | Value |
|---|---|
| Dataset | MNIST, 60,000 images, 28×28 grayscale |
| Split | 20,000 train / 5,000 validation / 5,000 test (seed 42, 30,000 unused) |
| Architecture | `784 → 1024 (ReLU) → 1024 (ReLU) → 1024 (ReLU) → 10` |
| Dropout | input `p = 0.2`, hidden `p = 0.5` (inverted dropout) |
| Optimizer | SGD, lr = 0.01, momentum = 0.95 |
| Batch size | 128 |
| Epochs | 20 |
| Loss | cross-entropy |

## Reference numbers from the paper (feed-forward MNIST nets)

| Model | Test error |
|---|---|
| Standard 2-layer net, 800 logistic units (Simard et al., 2003) | 1.60% |
| Dropout net, 3 layers, 1024 logistic units | 1.35% |
| Dropout net, 3 layers, 1024 ReLU units | **1.25%** |
| Dropout net + max-norm, 3 layers, 1024 ReLU units | 1.06% |

See `PAPER.md` for the full reproduction report.
