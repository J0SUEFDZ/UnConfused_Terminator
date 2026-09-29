# UnConfused Terminator

A neural network written from scratch in NumPy that learns to classify
handwritten digits from the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

It was a 2018 assignment for the Artificial Intelligence course in Computer
Engineering at the Instituto Tecnológico de Costa Rica. The goal was to
implement the pieces of a neural network by hand: linear layers, an activation
function, a loss function and backpropagation. No deep learning framework is
used.

It's kept here as it was submitted: a snapshot of coursework, not a maintained
project.

## Contents

| File | Description |
| --- | --- |
| [`UnConfused_Terminator___Josue_Fernandez___2013033195.py`](UnConfused_Terminator___Josue_Fernandez___2013033195.py) | The network and training loop. |
| [`UnConfused_Terminator___Josue_Fernandez___2013033195.pdf`](UnConfused_Terminator___Josue_Fernandez___2013033195.pdf) | The written report, in Spanish: methodology, experiments and conclusions. |

## How it works

- **Data:** the 70,000 MNIST images, each 28×28 pixels flattened to 784
  inputs, with their labels one-hot encoded into 10 classes (digits 0–9).
- **Architecture:** 784 inputs → one hidden layer of 10 units → 10 outputs.
  Weights use Xavier initialization. There are no bias terms.
- **Activation:** sigmoid. The report notes that ReLU was tried first but
  produced `NaN` losses right after the first batch.
- **Output and loss:** softmax with cross-entropy.
- **Training:** a single pass over the dataset in mini-batches (1,000 by
  default), with weights updated by backpropagation after each batch. Each
  batch's loss is printed as it trains.
- **Configuration:** hyperparameters are module-level variables at the top of
  the script (`layers`, `layersize_1`, `layersize_2`, `guardar` to pickle the
  weights to `save.p`).

## Findings from the report

- Smaller batches reached lower loss. The report gives values below 1 for a
  batch size of 10, about 20 for 100, and about 120 for 1,000.
- With small batches the loss stayed flat, then jumped sharply (for example
  from about 14 down to 0.000045, or back up). The report attributes this to
  the data not being shuffled. The dataset as loaded was ordered by label, so
  training saw all the 0s first, then all the 1s, and so on.

## Known limitations

This is an early learning exercise, and it has gaps:

- **No evaluation.** It never computes accuracy or runs predictions on
  held-out data. `train_test_split` is imported but unused, and
  `Predict_Image` is an empty stub.
- **Only one hidden layer is trained.** Setting `layers = 2` creates weights
  for a second hidden layer, but the training loop only handles one.
- **The backpropagation math has errors.** The gradient for `W2` uses the
  output pre-activation `z2` where it should use the hidden activation `a1`,
  and the sigmoid derivative is applied to the softmax output. The matrix
  shapes only line up because the hidden layer happens to have 10 units, the
  same as the number of classes.
- **No learning rate, shuffling or input normalization.** Pixel values stay in
  the 0–255 range.

## Running it

The script was written for Python 3 with NumPy, scikit-learn and matplotlib.
It loads MNIST with `sklearn.datasets.fetch_mldata`, which has since been
removed from scikit-learn, and its mldata.org data source is offline. To run
it today, replace the loading line with:

```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(int)
```

Then run:

```bash
pip install numpy scikit-learn matplotlib
python UnConfused_Terminator___Josue_Fernandez___2013033195.py
```

## References

- Shamdasani, S. (2017). [Build a flexible Neural Network with Backpropagation in Python](https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python).
- Britz, D. [nn-from-scratch](https://github.com/dennybritz/nn-from-scratch).
