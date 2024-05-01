# Torch

Basic implementation for the main modules of PyTorch. For educational purposes only. 

`main.py` contains an example of using the built modules to solve the logical AND gate problem.

`visualize.py` contains an example of visualizing the computational graph of the following equation:

$$Q = a + b \cdot c + d ^ e + \log_{10}(f)$$

![Computational Graph](graph.png)

# TODO

- [x] Implement `autograd` module to support arithmetic operations on scalars.
- [x] Simple implementation of `optim` and `nn` modules.
- [x] Visualizing the computational graph.
- [ ] N-dimensional arrays (Tensors) instead of scalars.
- [ ] Add Tensor operations.
- [ ] Implement `nn` module and `optim` to support Tensors and N-dimensional inputs.
