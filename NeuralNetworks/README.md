# Impelement Nueral Network

This project implements a simple neural network from scratch using Python, alongside a comparison with the Scikit-learn library. The project focuses on binary classification using the "moons" dataset, which is a common benchmark for evaluating classification algorithms.

## Overview

The project consists of two types of neural networks:
- A **Single Layer Neural Network** that implements forward and backward propagation manually.
- A **Multi-Layer Neural Network** that extends the single-layer implementation to include hidden layers.

Both implementations are compared against Scikit-learn's `MLPClassifier` to evaluate their performance.

## Installation

To set up the project, ensure you have Python installed on your machine. Then, install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

To run the project, execute the `main.py` file in the `src` directory. This will train both the manual implementations of the neural networks and the Scikit-learn model, and print their respective accuracies on the test set.

```bash
python src/main.py
```

## Files

- `src/__init__.py`: Marks the directory as a Python package.
- `src/data_preparation.py`: Contains code for generating and splitting the dataset.
- `src/single_layer_nn.py`: Defines the `SingleLayerNN` class with methods for training and prediction.
- `src/multi_layer_nn.py`: Defines the `MultiLayerNN` class with methods for training and prediction.
- `src/main.py`: Entry point for the project, orchestrating the training and evaluation of the models.
- `requirements.txt`: Lists the dependencies required for the project.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
