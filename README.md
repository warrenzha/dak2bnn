# Deep Additive Kernel (DAK)
This repository is a PyTorch implementation of **D**eep **A**dditive **K**ernel (DAK) model.

## Model
[Model architecture of Deep Additive Kernel (DAK).](assets/DAK.pdf)


## Usage
To reproduce the experiments, first install the required packages.
```bash
$ pip install -r requirement.txt
```

### Toy Example
Jupyter notebook for the toy example: [`examples/notebooks/2_DKL_example.ipynb`](examples/notebooks/2_DKL_example.ipynb)

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
1. Liang Ding, Rui Tuo, and Shahin Shahrampour. [A Sparse Expansion For Deep Gaussian Processes](https://www.tandfonline.com/doi/pdf/10.1080/24725854.2023.2210629). IISE Transactions (2023): 1-14.
2. Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, and Eric P. Xing. [Deep kernel learning." Artificial intelligence and statistics](https://proceedings.mlr.press/v51/wilson16.pdf). Artificial intelligence and statistics, pp. 370-378. PMLR, 2016.
3. Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, and Eric P. Xing. [Stochastic variational deep kernel learning](https://proceedings.neurips.cc/paper_files/paper/2016/file/bcc0d400288793e8bdcd7c19a8ac0c2b-Paper.pdf). Advances in neural information processing systems 29 (2016).