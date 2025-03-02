# Impelement Gradient Descent

This project implements three types of gradient descent algorithms: Standard Gradient Descent, Stochastic Gradient Descent, and Mini-Batch Gradient Descent. Each algorithm is implemented in its own Python file within the `src` directory.

## Project Structure

```
GradientDescent
├── src
│   ├── batch_gradient_descent.py      # Implementation of standard gradient descent
│   ├── stochastic_gradient_descent.py # Implementation of stochastic gradient descent
│   ├── mini_batch_gradient_descent.py # Implementation of mini-batch gradient descent
│   └── train.py                       # Entry point for training the models
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

## Installation

To set up the project, you need to have Python installed on your machine. You can then install the required dependencies using pip. Run the following command in your terminal:

```
pip install -r requirements.txt
```

## Usage

1. Navigate to the project directory:

   ```
   cd GradientDescent
   ```

2. Run the training script to execute the gradient descent algorithms:

   ```
   python src/train.py
   ```

This will generate synthetic data and apply the selected gradient descent algorithm to train the model.

## Algorithms

- **Batch Gradient Descent**: This algorithm updates the model parameters using the entire dataset, which can be computationally expensive for large datasets.

- **Stochastic Gradient Descent**: This algorithm updates the model parameters using one sample at a time, which allows for faster convergence but can introduce noise in the updates.

- **Mini-Batch Gradient Descent**: This algorithm updates the model parameters using a small batch of samples, balancing the benefits of both standard and stochastic gradient descent.

## Contributing

If you would like to contribute to this project, feel free to submit a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
