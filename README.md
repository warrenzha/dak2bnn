# Deep Additive Kernel (DAK)
This repository is a PyTorch implementation of **D**eep **A**dditive **K**ernel (DAK) model.

## Model
[Model architecture of Deep Additive Kernel (DAK).](assets/DAK.pdf)

## Benchmark
- NN
- NN + SVGP ([GPyTorch](https://docs.gpytorch.ai/en/v1.6.0/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html))
- SV-DKL ([GPyTorch](https://docs.gpytorch.ai/en/v1.6.0/examples/06_PyTorch_NN_Integration_DKL/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.html))
- AV-DKL (folder [`gpinfuser`](gpinfuser) is from the [repo](https://github.com/alanlsmatias/amortized-variational-dkl))

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
1. Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, and Eric P. Xing. [Deep kernel learning](https://proceedings.mlr.press/v51/wilson16.pdf). Artificial intelligence and statistics, pp. 370-378. PMLR, 2016.
2. Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, and Eric P. Xing. [Stochastic variational deep kernel learning](https://proceedings.neurips.cc/paper_files/paper/2016/file/bcc0d400288793e8bdcd7c19a8ac0c2b-Paper.pdf). Advances in neural information processing systems 29 (2016).
3. Alan L.S. Matias, César Lincoln Mattos, João Paulo Pordeus Gomes, and Diego Mesquita. [Amortized Variational Deep Kernel Learning](https://openreview.net/pdf?id=MSMKQuZhD5). Forty-first International Conference on Machine Learning.