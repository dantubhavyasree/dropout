# models

Trained model weights are stored here after running `notebooks/Training.ipynb`:

| File | Model |
|---|---|
| `mnist_nodropout.pt` | MLP trained **without** dropout |
| `mnist_dropout.pt` | MLP trained **with** dropout (p=0.5 hidden) |

Checkpoints are `.gitignore`d (`models/*.pt`) because they are large binaries that are
regenerated in Colab — do not commit them. If you want to share them anyway, use
[Git LFS](https://git-lfs.com/) or upload them to a release.