# Deep Learning with CNNs and RNNs

The repository covers convolution operations, image filtering and pooling, convolutional neural network architectures, recurrent neural networks for sentiment classification, and character-level text generation.


---

## Repository Contents

```text
.
├── CNN_Architectures.ipynb
├── Convolution_Operations .ipynb
├── Filters_Pooling.ipynb
├── README.md
├── RNN_Sentiment_Classification.ipynb
└── RNN_Text_Generation.ipynb
```

| Notebook | Main Topic | Key Concepts |
|---|---|---|
| [`CNN_Architectures.ipynb`](CNN_Architectures.ipynb) | CNN architecture design | AlexNet, convolution layers, dropout, residual blocks, skip connections, ResNet-style networks |
| [`Convolution_Operations .ipynb`](Convolution_Operations%20.ipynb) | Convolution fundamentals | Kernels, stride, VALID padding, SAME padding, manual padding |
| [`Filters_Pooling.ipynb`](Filters_Pooling.ipynb) | Image filtering and pooling | Sobel filters, edge detection, max pooling, average pooling |
| [`RNN_Sentiment_Classification.ipynb`](RNN_Sentiment_Classification.ipynb) | Sentiment classification | IMDB reviews, embeddings, LSTM, binary classification, confusion matrix |
| [`RNN_Text_Generation.ipynb`](RNN_Text_Generation.ipynb) | Character-level text generation | Tiny Shakespeare, sequence modeling, LSTM, sampling, temperature |

---

# 1. CNN Architectures

### `CNN_Architectures.ipynb`

This notebook demonstrates how larger convolutional neural networks can be constructed from basic PyTorch layers.

It contains two major implementations.

## AlexNet-Style CNN

The first model is an AlexNet-inspired architecture composed of:

- Five convolutional layers
- ReLU activation functions
- Three max-pooling operations
- Two large fully connected hidden layers
- 50% dropout for regularization
- A final 10-class output layer
- Softmax probabilities

The notebook also calculates the total number of trainable parameters.

### Saved notebook result

```text
Total Trainable Parameters: 24,767,882
```

The classifier is configured for a flattened feature map of `256 × 2 × 2`, so the input image dimensions need to produce that feature-map size after the convolution and pooling operations.

## ResNet-Style CNN

The notebook also introduces residual learning through a custom `ResidualBlock`.

Each residual block contains:

```text
Input
  │
  ├─────────────── Skip Connection ───────────────┐
  │                                               │
  ▼                                               │
3×3 Convolution → ReLU → 3×3 Convolution          │
  │                                               │
  └──────────────── Add ◄─────────────────────────┘
                          │
                         ReLU
```

The original input is added back to the transformed output, demonstrating the key idea behind **skip connections** in ResNet architectures.

The simplified ResNet-like model includes:

- One initial `7 × 7` convolution
- 64 output channels
- Stride of 2
- Two residual blocks
- A fully connected layer with 128 neurons
- A 10-class output layer

### Saved notebook result

```text
Total Trainable Parameters: 2,255,754
```

### Concepts demonstrated

- Building custom `nn.Module` classes
- Convolutional feature extraction
- Fully connected classification layers
- Dropout regularization
- Parameter counting
- Residual learning
- Identity/skip connections
- Model composition in PyTorch

---

# 2. Convolution Operations

### `Convolution_Operations .ipynb`

This notebook focuses on the mechanics of two-dimensional convolution.

A `5 × 5` input matrix is convolved with the following `3 × 3` Laplacian-style kernel:

```text
 0   1   0
 1  -4   1
 0   1   0
```

The example demonstrates four combinations of stride and padding:

| Case | Stride | Padding | Output Size |
|---|---:|---|---|
| 1 | 1 | VALID | `3 × 3` |
| 2 | 1 | SAME | `5 × 5` |
| 3 | 2 | VALID | `2 × 2` |
| 4 | 2 | SAME | `3 × 3` |

The notebook uses `torch.nn.functional.conv2d()` for the convolution operations.

For stride 2 with SAME padding, the notebook manually pads the input using:

```python
torch.nn.functional.pad()
```

This is useful for understanding that the dimensions of a convolution output are determined by the relationship between:

- Input size
- Kernel size
- Padding
- Stride

### Example output

For stride 1 with VALID padding, the selected input and Laplacian kernel produce:

```text
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
```

The zero values occur because the interior of the example matrix changes linearly, while the Laplacian operator responds to changes in the rate of intensity variation.

### Concepts demonstrated

- 2D convolution
- Kernel/filter construction
- Tensor reshaping for PyTorch
- Stride
- VALID padding
- SAME padding
- Manual padding
- Convolution output dimensions

---

# 3. Filters and Pooling

### `Filters_Pooling.ipynb`

This notebook contains two computer-vision exercises: **edge detection** and **pooling**.

## Sobel Edge Detection

A grayscale image is processed using Sobel filters.

### Sobel-X

```text
-1   0   1
-2   0   2
-1   0   1
```

Sobel-X emphasizes changes in the horizontal image direction and is commonly used to detect predominantly vertical edges.

### Sobel-Y

```text
-1  -2  -1
 0   0   0
 1   2   1
```

Sobel-Y emphasizes changes in the vertical image direction and is commonly used to detect predominantly horizontal edges.

The filters are applied using OpenCV's `filter2D()`, and the original and filtered images are visualized with Matplotlib.

> **Image requirement:** this notebook expects a local file named `parrot.jpg`. Place the image in the same directory as the notebook, or modify the image path in the code.

## Max Pooling and Average Pooling

The second part creates a random `4 × 4` tensor and applies:

- `nn.MaxPool2d(kernel_size=2, stride=2)`
- `nn.AvgPool2d(kernel_size=2, stride=2)`

Both operations reduce the spatial dimensions:

```text
4 × 4  →  2 × 2
```

### Max pooling

Selects the largest value in each pooling window, helping preserve strong activations.

### Average pooling

Calculates the mean of each pooling window, producing a smoother summary of the local region.

### Concepts demonstrated

- Image preprocessing
- Grayscale image representation
- Sobel edge detection
- Spatial filtering
- Feature-map downsampling
- Max pooling
- Average pooling
- Tensor dimensions


---

# 4. RNN Sentiment Classification

### `RNN_Sentiment_Classification.ipynb`

This notebook builds an LSTM-based binary sentiment classifier for movie reviews using the **IMDB dataset**.

The task is:

```text
Movie Review → LSTM → Negative or Positive
```

Labels are defined as:

```text
0 = Negative
1 = Positive
```

## Dataset

The IMDB dataset is loaded through TensorFlow/Keras with the vocabulary limited to the **10,000 most frequent words**.

The notebook uses:

- 25,000 original training reviews
- 25,000 test reviews
- 12,500 negative and 12,500 positive reviews in the original training set
- Maximum sequence length of 250 tokens

The original training set is further divided into:

```text
Training:   20,000 reviews
Validation:  5,000 reviews
Testing:    25,000 reviews
```

The split is stratified so that the sentiment classes remain balanced.

## Sequence Preprocessing

Reviews longer than 250 tokens are truncated, while shorter reviews are padded with zeros.

The notebook separately tracks the true sequence lengths so the classifier can use the LSTM output corresponding to the final real token instead of a padded position.

## Model Architecture

```text
Token IDs
   │
   ▼
Embedding Layer
10,000 words → 128-dimensional vectors
   │
   ▼
LSTM
128 hidden units
   │
   ▼
Dropout
p = 0.30
   │
   ▼
Linear Layer
128 → 1
   │
   ▼
Sigmoid during evaluation
   │
   ▼
Negative / Positive
```

Training uses:

- `BCEWithLogitsLoss`
- Adam optimizer
- Learning rate of `0.001`
- Batch size of `64`
- 8 epochs
- Gradient clipping with maximum norm `5`

## Saved Results

The saved notebook run reached:

```text
Test Accuracy: 0.8355
```

or approximately:

**83.55% test accuracy**

### Classification performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 0.8046 | 0.8861 | 0.8434 |
| Positive | 0.8733 | 0.7849 | 0.8267 |

### Confusion matrix

```text
                 Predicted
               Neg      Pos
Actual Neg    11076     1424
Actual Pos     2689     9811
```

The notebook also visualizes the confusion matrix and prints a full classification report.

> Results above come from the saved notebook execution. Training results may vary slightly when the notebook is rerun.

### Concepts demonstrated

- Natural language preprocessing
- Sequence padding and truncation
- Train/validation/test splitting
- Word embeddings
- LSTM sequence modeling
- Binary classification
- `BCEWithLogitsLoss`
- Gradient clipping
- Accuracy, precision, recall, and F1-score
- Confusion-matrix interpretation

---

# 5. RNN Text Generation

### `RNN_Text_Generation.ipynb`

This notebook creates a character-level LSTM that learns patterns from the **Tiny Shakespeare** dataset and generates new text one character at a time.

## Dataset

The notebook downloads the Tiny Shakespeare text dataset directly from GitHub.

The saved run contains:

```text
Number of characters: 1,115,394
Vocabulary size: 65
```

The vocabulary contains letters, punctuation, spaces, and newline characters.

Each character is mapped to an integer:

```text
Character → Integer ID → Embedding → LSTM
```

## Sequence Preparation

The dataset is divided into overlapping sequences of 100 characters.

For each input sequence, the target is the same sequence shifted one character forward.

Example conceptually:

```text
Input:   "HELLO"
Target:  "ELLO "
```

The saved notebook creates:

```text
1,115,294 training sequences
```

with a batch size of `64`.

## Model Architecture

```text
Character IDs
     │
     ▼
Embedding Layer
65 → 128
     │
     ▼
2-Layer LSTM
Hidden Size = 256
Dropout = 0.20
     │
     ▼
Linear Layer
256 → 65
     │
     ▼
Probability Distribution
over next character
```

The model uses:

- Character embedding size: `128`
- Hidden size: `256`
- LSTM layers: `2`
- Dropout: `0.20`
- Cross-entropy loss
- Adam optimizer
- Learning rate: `0.002`
- Gradient clipping
- 5 training epochs

The code automatically selects CUDA when available and otherwise uses the CPU.

## Training Results

The saved run shows the loss decreasing during training:

| Epoch | Loss |
|---:|---:|
| 1 | 1.2297 |
| 2 | 1.1221 |
| 3 | 1.1059 |
| 4 | 1.0986 |
| 5 | 1.0949 |

## Temperature-Based Text Generation

The notebook generates text beginning with:

```text
ROMEO:
```

and compares different temperature settings:

- **0.3** — more conservative and predictable output
- **0.8** — a balance between structure and variation
- **1.5** — more random and experimental output

Temperature controls the probability distribution used when sampling the next character.

In general:

```text
Lower temperature  → safer, more repetitive choices
Higher temperature → more diverse, less predictable choices
```

This demonstrates how the same trained neural network can produce very different outputs depending on the sampling strategy.

### Concepts demonstrated

- Character-level language modeling
- Vocabulary construction
- Character encoding and decoding
- Custom PyTorch `Dataset`
- Sequential training examples
- Embeddings
- Multi-layer LSTM networks
- Cross-entropy loss
- Gradient clipping
- Autoregressive generation
- Temperature sampling
- GPU/CPU device selection

---

# Technologies Used

The notebooks use the following tools and libraries:

- **Python**
- **Jupyter Notebook**
- **PyTorch**
- **NumPy**
- **Matplotlib**
- **OpenCV**
- **Scikit-learn**
- **TensorFlow/Keras** — used to load the IMDB dataset
- **urllib** — used to download Tiny Shakespeare

---

# Installation

## 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

## 2. Create a virtual environment

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install the required packages

```bash
pip install jupyter torch torchvision numpy matplotlib opencv-python scikit-learn tensorflow
```

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Then open any notebook from the browser interface.


---

# Notes

These notebooks are intended primarily for **learning, experimentation, and demonstration**. They emphasize readable implementations of important deep learning ideas rather than production-level optimization.

Model performance should therefore be interpreted as an educational benchmark rather than as a state-of-the-art result.


