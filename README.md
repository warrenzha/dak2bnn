# Deep Additive Kernel (DAK)
This repository is a PyTorch implementation of **D**eep **A**dditive **K**ernel (DAK) model in "From Deep Additive Kernel Learning to Last-Layer Bayesian Neural
Networks via Induced Prior Approximation".

## Model
[Model architecture of Deep Additive Kernel (DAK).](assets/DAK.pdf)


## Usage
To reproduce the experiments, first install the required packages.
```bash
$ pip install -r requirement.txt
```

### Toy Example
The codes for the toy example are in [`examples/notebooks/5_GP_toy_example.ipynb`](https://github.com/hchen19/dak/blob/main/examples/notebooks/5_GP_toy_example.ipynb)

### UCI Regression
```bash
$ cd examples/uci
$ python run_uci.py 
```

### Image Classification
- [MNIST](https://yann.lecun.com/exdb/mnist/)
```bash
$ cd examples/mnist
$ python run_mnist.py 
```

- [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)
```bash
$ cd examples/cifar
$ python run_cifar.py 
```


## References
1. Liang Ding, Rui Tuo, and Shahin Shahrampour. [A Sparse Expansion For Deep Gaussian Processes](https://www.tandfonline.com/doi/pdf/10.1080/24725854.2023.2210629). IISE Transactions (2023): 1-14. [Code](https://github.com/ldingaa/DGP_Sparse_Expansion) in MATLAB version.
2. Rishabh Agarwal, et al. [Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://proceedings.neurips.cc/paper/2021/file/251bd0442dfcc53b5a761e050f8022b8-Paper.pdf). Advances in neural information processing systems 34 (2021): 4699-4711.
3. Wei Zhang, Brian Barr, and John Paisley. [Gaussian Process Neural Additive Models](https://arxiv.org/pdf/2402.12518.pdf). AAAI Conference on Artificial Intelligence (2024)
4. 