# Deep Additive Kernel (DAK)
This repository is a PyTorch implementation of **D**eep **A**dditive **K**ernel (DAK) model.

## Model
[Model architecture of Deep Additive Kernel (DAK).](assets/DAK.pdf)

## Benchmark
- DNN
- NN+SVGP ([GPyTorch](https://docs.gpytorch.ai/en/v1.6.0/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html))
- SV-DKL ([GPyTorch](https://docs.gpytorch.ai/en/v1.6.0/examples/06_PyTorch_NN_Integration_DKL/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.html))
- AV-DKL ([`gpinfuser`](gpinfuser) from this [repo](https://github.com/alanlsmatias/amortized-variational-dkl))
- DAK (package [`dak`](dak))

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

## Citation
If you find our work relevant to your research, please cite:
```bibtex
@article{
    title={From Deep Additive Kernel Learning to Last-Layer Bayesian Neural Networks via Induced Prior Approximation},
    author={Wenyuan Zhao and Haoyuan Chen and Tie Liu and Rui Tuo and Chao Tian},
    year={2025},
    journal={2025 International Conference on Artificial Intelligence and Statistics}
}
```