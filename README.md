# Handwritten Digit Recognition on MNIST: Perceptron vs. ANN vs. CNN

Three neural network architectures trained on the MNIST handwritten-digit dataset and compared on the same held-out test set, to show how much model design matters for image data.

![Accuracy comparison](images/accuracy_comparison.png)

## Results

All models: 5 epochs, batch size 32, trained on 60,000 images and scored on the separate 10,000-image test set (single run, seed 42).

| Model | Architecture | Optimizer | Parameters | Test accuracy |
|---|---|---|---|---|
| Perceptron | Flatten → Dense(10, softmax) | SGD | 7,850 | **90.89%** |
| ANN | Flatten → Dense(128) → Dense(64) → Dense(10) | Adam | 109,386 | **97.48%** |
| CNN | Conv(32) → Pool → Conv(64) → Pool → Dense(128) → Dropout(0.5) → Dense(10) | Adam | 225,034 | **99.09%** |

**Takeaways**
- The CNN improves on the single-layer perceptron by about 8 points and on the dense ANN by about 1.6 points.
- Flattening an image discards spatial structure; convolutions keep it, which is why the CNN wins with a modest parameter count.
- The ANN's training accuracy (98.8%) is noticeably above its test accuracy (97.5%), a sign of overfitting that the CNN's dropout layer helps limit.

## Visuals

| Training curves | CNN confusion matrix |
|---|---|
| ![Training curves](images/training_curves.png) | ![CNN confusion matrix](images/confusion_matrix_cnn.png) |

## Data

MNIST in CSV form: one `label` column (0-9) plus 784 pixel columns (28×28 grayscale, values 0-255).

- `mnist_train.csv`: 60,000 rows
- `mnist_test.csv`: 10,000 rows

The CSVs are not stored in this repo (the training file exceeds GitHub's 100 MB limit). Download them from a MNIST-in-CSV source such as Kaggle and place them next to the notebook.

## Pipeline

1. Load train/test CSVs with pandas
2. Scale pixels to [0, 1], reshape to 28×28 (28×28×1 for the CNN), one-hot encode labels
3. Train the three models with Keras
4. Compare accuracy/loss curves, side-by-side predictions, and the CNN confusion matrix

## Run it

```bash
pip install -r requirements.txt
jupyter notebook mnist_classification.ipynb
```

## Tech stack

Python · TensorFlow/Keras · scikit-learn · pandas · NumPy · Matplotlib · Seaborn

## Possible next steps

- Data augmentation (small shifts/rotations) and more epochs
- Learning-rate scheduling and early stopping
- Multiple seeds to report mean ± standard deviation
- Compare with classical baselines (logistic regression, SVM, random forest)
