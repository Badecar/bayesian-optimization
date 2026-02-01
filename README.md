# Bayesian Optimization for CNN Hyperparameters

This repository contains a framework for performing **Bayesian Optimization (BO)** to automate the hyperparameter tuning of Convolutional Neural Networks (CNNs).

Developed as part of the *Active Machine Learning and Agency* (02463) course at the Technical University of Denmark (DTU), this project demonstrates the efficiency of Gaussian Process-based optimization over standard grid or random search methods.

## 🧠 Project Overview

Training deep learning models requires selecting optimal hyperparameters—a process that is often computationally expensive and non-convex. This project solves that problem by treating the model training process as a "black-box function" and optimizing it using a probabilistic surrogate model.

### Core Components
* **Objective Function**: The validation accuracy of a CNN trained on the MNIST dataset.
* **Surrogate Model**: A Gaussian Process (GP) that models the probability distribution of the objective function.
* **Acquisition Function**: Uses the `gp_hedge` strategy (probabilistically choosing between EI, PI, and LCB) to balance exploration (searching new areas) and exploitation (refining known good areas).

## ⚙️ Search Space Configuration

The optimizer explores a complex, high-dimensional mixed-integer space to find the optimal architecture:

| Hyperparameter | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `conv_nodes_1` | Integer | 8 - 48 | Output channels for 1st Conv layer |
| `conv_nodes_2` | Integer | 8 - 48 | Output channels for 2nd Conv layer |
| `kernel_size_1` | Categorical | 3, 5 | Filter size for 1st Conv layer |
| `kernel_size_2` | Categorical | 3, 5 | Filter size for 2nd Conv layer |
| `maxpool_size` | Integer | 2 - 4 | Stride/Kernel size for pooling |
| `dropout_rate` | Categorical | 0.1 - 0.7 | Probability of neuron dropout |
| `fc_nodes` | Integer | 32 - 512 | Neurons in the fully connected layer |

## 🛠 Technology Stack

* **PyTorch**: For building and training the CNN architecture.
* **Scikit-Optimize (`skopt`)**: Implementation of the Bayesian Optimization engine (Gaussian Processes and minimization loop).
* **NumPy & Pandas**: Data manipulation and result logging.
* **Matplotlib**: Visualization of the optimization landscape and convergence.

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ installed.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/CarlSvejstrup/bayesian-optimization.git](https://github.com/CarlSvejstrup/bayesian-optimization.git)
    cd bayesian-optimization
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Optimizer
To start the optimization loop (default: 45 iterations with 5 initial random points):

```bash
python main.py
