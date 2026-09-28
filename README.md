# Deep Learning Foundations with PyTorch

This repository contains a collection of Jupyter notebooks exploring core concepts in **deep learning**, with an emphasis on **convolutional neural networks (CNNs)**, **convolution operations**, **image filters and pooling**, and **recurrent neural networks (RNNs/LSTMs)** using PyTorch.

The notebooks move from low-level operations such as convolution, padding, stride, filtering, and pooling to higher-level neural network architectures and sequence modeling.

## Repository Structure

```text
.
├── CNN_Architectures.ipynb
├── Convolution_Operations .ipynb
├── Filters_Pooling.ipynb
├── README.md
├── RNN_for_Text_generation.ipynb
└── Sentiment_Classification_Using_RNN.ipynb
```

## Notebook Overview

### 1. `CNN_Architectures.ipynb`

Explores important CNN architecture concepts using PyTorch.

Topics include:

- Building an **AlexNet-style convolutional neural network**
- Convolutional and max-pooling layers
- Fully connected layers and dropout
- Softmax classification output
- Counting trainable model parameters
- Building a **residual block**
- Skip connections
- Constructing a simple **ResNet-like architecture**

This notebook demonstrates how increasingly complex CNN architectures can be built from standard PyTorch layers.

---

### 2. `Convolution_Operations .ipynb`

Demonstrates the mechanics of two-dimensional convolution using a small numerical example.

The notebook uses:

- A `5 × 5` input matrix
- A `3 × 3` Laplacian-style kernel
- PyTorch `conv2d`
- Different stride values
- `VALID` and `SAME` padding

The following convolution cases are demonstrated:

| Stride | Padding |
|---|---|
| 1 | VALID |
| 1 | SAME |
| 2 | VALID |
| 2 | SAME |

Because PyTorch does not support `padding="same"` with a stride greater than 1 in this example, the notebook also demonstrates **manual padding** with `torch.nn.functional.pad()`.

---

### 3. `Filters_Pooling.ipynb`

Explores image filtering and spatial pooling.

#### Image filtering

The notebook:

- Loads a grayscale image with OpenCV
- Defines **Sobel-X** and **Sobel-Y** filters
- Applies the filters to detect horizontal and vertical intensity changes
- Visualizes the original image and detected edges with Matplotlib

#### Pooling

The notebook also demonstrates:

- `MaxPool2d`
- `AvgPool2d`
- A randomly generated `4 × 4` input matrix
- Reduction of the feature map using a `2 × 2` pooling window with stride 2

> **Note:** The filtering section expects a file named `parrot.jpg` to be available in the same working directory as the notebook.

---

### 4. `RNN_for_Text_generation.ipynb`

Implements a character-level text generation model using an **LSTM recurrent neural network**.

The notebook:

- Downloads the **Tiny Shakespeare** dataset
- Identifies the character vocabulary
- Converts characters to integer representations
- Creates input-target sequences for next-character prediction
- Builds a custom PyTorch `Dataset`
- Uses an embedding layer
- Implements a multi-layer **LSTM**
- Trains the model with cross-entropy loss and the Adam optimizer
- Uses gradient clipping to reduce exploding gradients
- Generates new text from a starting prompt
- Compares text generation at different temperature values

Example generation begins with:

```text
ROMEO:
```

The notebook demonstrates how sampling temperature affects the randomness and creativity of generated text.

---

### 5. `Sentiment_Classification_Using_RNN.ipynb`

The current contents of this notebook focus on **image classification rather than sentiment analysis**.

It currently includes:

- Loading the **Fashion-MNIST** dataset from OpenML
- Normalizing and reshaping image data
- Creating PyTorch `DataLoader` objects
- Building a multi-layer CNN
- Training and evaluation functions
- Cross-entropy loss
- Adam optimization
- Dropout regularization
- A residual unit for a ResNet-style architecture

> **Repository note:** The filename does not currently match the notebook contents. If the current implementation is intentional, consider renaming this notebook to something such as `Fashion_MNIST_CNN_ResNet.ipynb`. If sentiment classification is the intended topic, the notebook contents can instead be replaced with an RNN/LSTM-based sentiment model.

## Technologies Used

- Python
- Jupyter Notebook
- PyTorch
- Torchvision
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

## Installation

Clone the repository and move into the project directory:

```bash
git clone <your-repository-url>
cd <repository-folder>
```

Create and activate a virtual environment if desired, then install the main dependencies:

```bash
pip install torch torchvision numpy matplotlib opencv-python scikit-learn jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open the notebooks in the order shown in the repository or explore them individually.

## Data and External Resources

Some notebooks use external data or local files:

- `Filters_Pooling.ipynb` requires a local image named `parrot.jpg`.
- `RNN_for_Text_generation.ipynb` downloads the Tiny Shakespeare dataset when executed.
- `Sentiment_Classification_Using_RNN.ipynb` retrieves Fashion-MNIST through `sklearn.datasets.fetch_openml`.

An internet connection is therefore required when running notebooks that download datasets.

## Hardware Notes

`RNN_for_Text_generation.ipynb` automatically uses Apple Metal Performance Shaders (**MPS**) when available and otherwise falls back to the CPU.

The current Fashion-MNIST notebook explicitly moves the model and tensors to `"mps"`. If you are running the notebook on a system without Apple MPS support, update the device configuration to use CPU or CUDA as appropriate.

A portable approach is:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
```

Then move the model and tensors with:

```python
model = model.to(device)
inputs = inputs.to(device)
targets = targets.to(device)
```

## Learning Objectives

By working through this repository, you can practice how to:

- Explain convolution, kernels, stride, and padding
- Apply image filters for edge detection
- Compare max pooling and average pooling
- Build CNN architectures in PyTorch
- Understand residual connections and ResNet-style models
- Prepare sequential text data for neural networks
- Build and train an LSTM for character-level text generation
- Use temperature-based sampling for generated text
- Create training and evaluation workflows with PyTorch

## Suggested Notebook Order

For a progressive learning path:

1. `Convolution_Operations .ipynb`
2. `Filters_Pooling.ipynb`
3. `CNN_Architectures.ipynb`
4. `RNN_for_Text_generation.ipynb`
5. `Sentiment_Classification_Using_RNN.ipynb`

This order begins with the mathematical building blocks of CNNs, progresses into complete CNN architectures, and then introduces recurrent neural networks and sequence modeling.

## Purpose

This repository serves as a practical collection of deep-learning exercises demonstrating how fundamental neural-network concepts can be implemented directly with Python and PyTorch.
