import copy
import math
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

# Shorthand:
# "pd_" as a variable prefix means "partial derivative"
# "d_" as a variable prefix means "derivative"
# "_wrt_" is shorthand for "with respect to"
# "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and input to hidden layer neurons respectively


class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(
        self,
        num_inputs,
        num_hidden,
        num_outputs,
        hidden_layer_weights=None,
        hidden_layer_bias=None,
        output_layer_weights=None,
        output_layer_bias=None,
    ):
        self.num_inputs = num_inputs
        self.epochs = 300
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(
            output_layer_weights
        )
        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        index = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(
                        hidden_layer_weights[index]
                    )
                index += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(
        self, output_layer_weights
    ):
        index = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(
                        output_layer_weights[index]
                    )
                index += 1

    def inspect(self):
        print("------")
        print("* Inputs: {}".format(self.num_inputs))
        print("------")
        print("Hidden Layer")
        self.hidden_layer.inspect()
        print("------")
        print("* Output Layer")
        self.output_layer.inspect()
        print("------")

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        output_layer_outputs = self.output_layer.feed_forward(hidden_layer_outputs)
        return output_layer_outputs

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. Output neuron deltas
        # Your Code Here
        # ∂E/∂zⱼ

        # 2. Hidden neuron deltas
        # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
        # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
        # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
        # Your Code Here

        # 3. Update output neuron weights
        # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
        # Δw = α * ∂Eⱼ/∂wᵢ
        # Your Code Here

        # 4. Update hidden neuron weights
        # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
        # Δw = α * ∂Eⱼ/∂wᵢ
        # Your Code Here


    def calculate_total_error(self, training_sets):
        # Your Code Here
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(
                    training_outputs[o]
                )
        return total_error

    def plotInfo(self, training_sets):
        x = list(range(0, self.epochs))
        plt.figure(1)

        plt.xlabel("epochs")
        plt.ylabel("loss")
        loss_epochs = []
        for i in range(self.epochs):
            for k in range(len(training_sets)):
                inputs, outputs = training_sets[k]
                self.train(inputs, outputs)
            loss = self.calculate_total_error(training_sets)
            loss_epochs.append(loss / len(training_sets))
        plt.plot(x, loss_epochs, color="red", linewidth=0.5)
        # plt.show()

        plt.figure(2)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")

        accuracy_epochs = []

        for i in range(self.epochs):
            accuracy = 0
            for i in range(len(training_sets)):
                inputs, outputs = training_sets[i]
                self.train(inputs, outputs)
                neuron_output = self.feed_forward(inputs)
                if abs(neuron_output[0] - outputs[0]) < 0.01:
                    accuracy += 1
            accuracy_rate = accuracy / len(training_sets)
            accuracy_epochs.append(accuracy_rate)

        plt.plot(x, accuracy_epochs, color="blue", linewidth=0.5)
        plt.show()


class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print("Neurons:", len(self.neurons))
        for n in range(len(self.neurons)):
            print(" Neuron", n)
            for w in range(len(self.neurons[n].weights)):
                print("  Weight:", self.neurons[n].weights[w])
            print("  Bias:", self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total_net_input = self.bias
        for i in range(len(self.inputs)):
            total_net_input += self.inputs[i] * self.weights[i]
        return total_net_input

    # Apply the logistic function to squash the output of the neuron
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    def squash(self, total_net_input):
        # sigmoid 函数
        return 1 / (1 + math.exp(-total_net_input))

    # Determine how much the neuron's total input has to change to move closer to the expected output
    #
    # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    #
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return (
            self.calculate_pd_error_wrt_output(target_output)
            * self.calculate_pd_total_net_input_wrt_input()
        )

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    #
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # = actual output - target output
    #
    # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    #
    # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    # Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
    #
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]


def loadData(filename):

    dataSet = []
    with open(filename) as fr:
        for i, line in enumerate(fr.readlines()):
            cur_line = []
            now_line = line.strip("\n").strip(" ").split(" ")
            for j in range(len(now_line)):
                if now_line[j] == "?":
                    now_line[j] = -1
            result_line = list(map(float, now_line[23]))
            now_line = list(map(float, now_line[:22]))
            cur_line.append(now_line)
            cur_line.append(result_line)
            dataSet.append(cur_line)
    return dataSet


def main():
    train_data = loadData("horse-colic.data")
    test_data = loadData("horse-colic.test")

    nn = NeuralNetwork(len(train_data[0][0]), 5, len(train_data[0][1]))
    for i in range(100):
        training_inputs, training_outputs = random.choice(train_data)
        nn.train(training_inputs, training_outputs)

    # print(nn.inspect())
    accuracy = 0
    for i in range(len(test_data)):
        training_inputs, training_outputs = test_data[i]
        nn.train(training_inputs, training_outputs)
        neuron_output = nn.feed_forward(training_inputs)
        if abs(neuron_output[0] - training_outputs[0]) < 0.01:
            accuracy += 1
    accuracy_rate = accuracy / len(test_data)
    print("the accuracy rate is", accuracy_rate)

    nn.plotInfo(test_data)


if __name__ == "__main__":
    main()
