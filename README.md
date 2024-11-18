# AI Neural Network Framework

An AI neural network framework implemented in Go, designed for building flexible and customizable neural network architectures. The framework supports various neuron types and allows for the construction of complex networks by defining neuron configurations via JSON.

## Overview

This framework provides a way to build neural networks with a variety of neuron types, including:

- **Input Neurons**: Receive input data to the network.
- **Dense (Fully Connected) Neurons**: Standard neurons connected to all neurons in the previous layer.
- **Recurrent Neurons (RNN)**: Neurons that maintain state across time steps, suitable for sequential data.
- **Long Short-Term Memory Neurons (LSTM)**: Advanced recurrent neurons that can learn long-term dependencies.
- **Convolutional Neurons (CNN)**: Neurons that apply convolution operations, commonly used in image processing.
- **Attention Mechanism Neurons**: Neurons that implement attention mechanisms to focus on specific parts of the input.
- **Output Neurons**: Provide the final output of the network.

## Features

- **Modular Design**: Organized into packages and files for clarity and maintainability.
- **Customizable Neurons**: Define neurons with specific types, biases, activation functions, and connections.
- **JSON Configuration**: Load neuron configurations from JSON, making it easy to modify network architectures without changing code.
- **Activation Functions**: Supports various activation functions, including ReLU, Sigmoid, Tanh, Leaky ReLU, ELU, and Linear.
- **Forward Propagation**: Implements forward propagation through the network, processing inputs through connected neurons.
- **Support for Recurrent Networks**: Handles timesteps for recurrent neurons like RNNs and LSTMs.
- **Attention Mechanism**: Includes an attention neuron type for implementing attention-based models.

## How It Works

1. **Neuron Definition**: Neurons are defined with unique IDs, types, biases, activation functions, and connections to other neurons.
2. **Connections**: Each neuron can connect to other neurons, with specified weights for each connection.
3. **Activation Functions**: Neuron outputs are computed using the specified activation functions.
4. **Forward Propagation**: Inputs are propagated through the network according to the connections and neuron types.
5. **Timesteps**: For recurrent neurons, the network processes inputs over multiple timesteps, maintaining state where appropriate.

## Getting Started

1. **Setup**: Ensure you have Go installed and set up a Go module for your project.
2. **Structure**: Organize your project with the framework files and your main application file.
3. **Define Network**: Create a JSON configuration that defines your network's neurons and their connections.
4. **Initialize**: Use the framework to load your neuron configurations and initialize the network.
5. **Run**: Provide inputs to the network and run it over the desired number of timesteps.
6. **Output**: Retrieve and utilize the outputs from the network's output neurons.

## Future Work

- **Training Capabilities**: Implement backpropagation and optimization algorithms to enable network training.
- **Advanced Features**: Add support for additional neuron types, such as GRUs or custom layers.
- **Performance Optimization**: Improve computational efficiency and support for larger networks.
- **Concurrency**: Leverage Go's concurrency features for parallel processing.

## Contributing

Contributions are welcome. Please open issues or submit pull requests to improve the framework.

## License

This project is licensed under the Apache License 2.0.
