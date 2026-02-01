# Bayesian Optimization

## Overview
Bayesian Optimization is a powerful technique for optimizing black-box functions that are expensive to evaluate. This repository is dedicated to comparing different acquisition functions used in Bayesian Optimization. Acquisition functions guide the optimization process by determining the next most informative data point to evaluate.

## Objectives
This project aims to:

- Explore and analyze various acquisition functions employed in Bayesian Optimization.
- Compare their performance across different benchmark problems.
- Provide insights into selecting appropriate acquisition functions for diverse applications.

## Features
- Implementation of commonly used acquisition functions such as:
  - **Expected Improvement (EI)**
  - **Upper Confidence Bound (UCB)**
  - **Probability of Improvement (PI)**
- Benchmark suite for testing optimization performance.
- Visualizations to contrast and evaluate the behavior of acquisition functions.

## Use Case
This repository is beneficial for practitioners, researchers, and students who want to:
- Understand how Bayesian Optimization works.
- Learn how acquisition functions impact optimization outcomes.
- Apply Bayesian Optimization to real-world black-box optimization tasks.

## Prerequisites
To use this codebase, ensure you have the following installed:
- Python 3.8 or higher
- Dependencies specified in `requirements.txt`.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/CarlSvejstrup/bayesian-optimization.git
    ```
2. Navigate to the project directory:
    ```bash
    cd bayesian-optimization
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Run the main script to start experimenting with different acquisition functions:
```bash
python main.py
```

Adjust parameters and settings in the configuration file (`config.yaml`) to customize your experiments.

---

This project was completed as part of the coursework for 02463 Active machine learning and agency at the Technical University of Denmark (DTU).
