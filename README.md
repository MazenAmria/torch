# Torch

Basic implementation for the main modules of PyTorch. For educational purposes only. 

## Setup

It's recommended to create a dedicated conda environment for this project.

```bash
(torch) $ sudo apt install libgraphviz-dev graphviz

(torch) $ pip install -r requirements.txt
```

## Training Sample Models

`and.py` contains an example of using the built modules to solve the logical AND gate problem. The model is a single neuron with sigmoid activation function.

```bash
(torch) $ python and.py

X = [0.0, 0.0], y_p = 0.001, y = 0.0
X = [0.0, 1.0], y_p = 0.074, y = 0.0
X = [1.0, 0.0], y_p = 0.074, y = 0.0
X = [1.0, 1.0], y_p = 0.912, y = 1.0

```

`xor.py` contains a network that solves the logical XOR gate problem.

```bash
(torch) $ python xor.py

X = [0.0, 0.0], y_p = 0.036, y = 0.0
X = [0.0, 1.0], y_p = 0.968, y = 1.0
X = [1.0, 0.0], y_p = 0.968, y = 1.0
X = [1.0, 1.0], y_p = 0.034, y = 0.0

```

## Visualizing the Computational Graph

`visualize.py` contains an example of visualizing the computational graph of the following equation:

$$Q = a + b \cdot c + d ^ e + \log_{g}(f)$$

![Computational Graph](graph.png)

# TODO

- [x] Implement `autograd` module to support arithmetic operations on scalars.
- [x] Simple implementation of `optim` and `nn` modules.
- [x] Visualizing the computational graph.
- [ ] N-dimensional arrays (Tensors) instead of scalars.
- [ ] Add Tensor operations.
- [ ] Implement `nn` module and `optim` to support Tensors and N-dimensional inputs.

# Note

The entire codebase has been written using vanilla VIM, with no plugins, no configurations, no Intellisense, no code completion, and no Copilot. The project was developed without any direct access to any of the existing `autograd` implementations (`pytorch`, `tinygrad`, `micrograd`, etc.), except for PyTorch API which the project is trying to re-implement. More importantly, only official documentation and forums (like Stackoverflow and Reddit) are allowed to be accessed while developing the project, and only for specific technical questions (i.e., questions that are not directly related to `autograd` implementation). ChatGPT or any other AI tool is prohibited.

This method turned out to be very helpful. It forces you to think and learn more. You'll be surprised by how much you can code without searching for anything or accessing any external documentation/forum.
