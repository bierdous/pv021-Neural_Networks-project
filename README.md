# PV021 project -- Deep Learning from Scratch

Authors:

Marek Dohnal (552405), Martin Bertko (514588)

## Description
- Implementation of a multi-layer perceptron in C++ without the use of advanced libraries or frameworks.
- Trained on the Fashion-MNIST dataset using a backpropagation
  algorithm.
- Accuracy of classification at least 88%

## Requirements
- A linux based system
- C++17

## How to run

NOTE: The network was evaluated on a multi-core server CPU with a runtime (parsing, training, inference, export) of 8-10 minutes. On a personal computer the runtime may be a lot longer.

To run parsing, training, inference, and export:

```
run.sh
```

Evaluation is ran according to a README file in the evaulator folder.
