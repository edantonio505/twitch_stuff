import math
import tqdm
import random
from typing import List


# ======================
# Linear Algebra
# ======================

Vector = List[float]

def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]




def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]




def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))




def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]












# ======================
# Calculo
# ======================
# sum of squares 
def sum_of_squares(v: Vector) -> float:
    """Computes the sum of squared elements in v"""
    return dot(v, v)





# Loss function
# returns error distance
def squared_distance(v: Vector, w: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))







# Gradient step downwards
# Gradient Descent
# performs a gradient step
def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)















# =================================
# NEURAL NETWORK
# =================================



# sigmoid function
def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t))


# Neuron ouput
def neuron_output(weights: Vector, inputs: Vector) -> float:
    # weights includes the bias term, inputs includes a 1
    output = sigmoid(dot(weights, inputs))
    return output




# feed forward algorithm
def feed_forward(neural_network: List[List[Vector]],
                 input_vector: Vector) -> List[Vector]:
    """
    Feeds the input vector through the neural network.
    Returns the outputs of all layers (not just the last one).
    """
    outputs: List[Vector] = []



    for layer in neural_network:

        input_with_bias = input_vector + [1]              # Add a constant.
        output = [neuron_output(neuron, input_with_bias)  # Compute the output
                  for neuron in layer]                    # for each neuron.
        outputs.append(output)                            # Add to results.

        # Then the input to the next layer is the output of this one
        input_vector = output
    return outputs









# Calculo + Feed_forward
# Finds the Gradient
# Calculus
def sqerror_gradients(network: List[List[Vector]],
                      input_vector: Vector,
                      target_vector: Vector) -> List[List[Vector]]:
    """
    Given a neural network, an input vector
    loss with respect to the neuron weights.r, and a target vector,
    make a prediction and compute the gradient of the squared error
    """
    # forward
    # feed forward
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # gradients with respect to output neuron pre-activation outputs
    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, target_vector)]

    # gradients with respect to output neuron weights
    output_grads = [[output_deltas[i] * hidden_output
                     for hidden_output in hidden_outputs + [1]]
                    for i, output_neuron in enumerate(network[-1])]

    # gradients with respect to hidden neuron pre-activation outputs
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                         dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # gradients with respect to hidden neuron weights
    hidden_grads = [[hidden_deltas[i] * input for input in input_vector + [1]]
                    for i, hidden_neuron in enumerate(network[0])]

    return [hidden_grads, output_grads]














# Regresa con el inde del valor mas grande dentro del Vector
# el index representa el numero predicho
def argmax(xs: list) -> int:
    """Returns the index of the largest value"""
    return max(range(len(xs)), key=lambda i: xs[i])








# convert MNIST from string to matrix of 
# vectors
def convert_to_float_vector(dataframe):
    new_dataframe = []
    new_label = []
    for vector in dataframe:
        all_data = vector.split(',')
        new_label.append(all_data[0])
        data = all_data[1:]    
        # rescale de data input number from ranges between 0.01 and 1.00
        new_dataframe.append([(float(number) / 255.0 * 0.99 + 0.01) for number in data])
    # create a label encoded thing
    encoded_labels = []
    for number in new_label:
        label = [0.1 for x in range(10)]
        label[int(number)] = 0.99
        encoded_labels.append(label)
    return encoded_labels, new_dataframe







# function that displays a number on the 
# 
def display_number(vector, label):
    line = ""
    message = ""
    linecount = 0
    for i in range(len(vector)):
        number = vector[i]
        if number > 0.01:
            number = "#"
        else:
            number = " "
        message += number
        if linecount == 28:
            message += line+"\n"
            line = ""
            linecount = 0 
        linecount += 1
    print("\nEl numero es {} ".format(argmax(label)))
    print(message)













def main():
    random.seed(0)
    # open file 
    data_file = open('mnist_dataset/mnist_train_100.csv', 'r')
    data_list = data_file.readlines()
    data_file.close()

    # matrix of handwritten digits
    # it has to also return a label vector
    y_train, X_train = convert_to_float_vector(data_list)


    # demostrando de 
    # el set de entrenamiento
    # un numero aleatorio para demostrar como funciona
    index_number = 7
    display_number(X_train[index_number], y_train[index_number])

    


    # HyperParameters
    # ================
    hidden_nodes = 100
    input_nodes = 784
    output_nodes = 10
    learning_rate = 0.2
    epochs = 30




    # Weights and network inicialization
    # create a layers 
    # matrices of weights for both the input 
    # and the hidden to output
    network = [
        # hidden layer: 784 inputs -> 100 outputs
        [[random.random()  / 100 for _ in range(input_nodes + 1)] for _ in range(hidden_nodes)],
    
        # output_layer: 100 inputs -> 10 outputs
        [[random.random() / 100 for _ in range(hidden_nodes + 1)] for _ in range(output_nodes)]
    ]





    
    
    
    # Neural Network 
    # Training
    with tqdm.trange(epochs) as t:
        for epoch in t:
            epoch_loss = 0.0
    
            for x, y in zip(X_train, y_train):

                # backpropagate
                # to measure loss function
                predicted = feed_forward(network, x)[-1]
                epoch_loss += squared_distance(predicted, y)
                # ======================================

                # backpropagation algorithm
                # output = sigmoid(W.X+b)
                # feed forward + gradient calculation 
                
                # calculate gradients
                gradients = sqerror_gradients(network, x, y)
    
                # Take a gradient step for each neuron in each layer
                # update the network by learning rate
                # gradient descent and updating weights of the algorithm
                network = [[  gradient_step(neuron, grad, -learning_rate)
                            for neuron, grad in zip(layer, layer_grad)]
                        for layer, layer_grad in zip(network, gradients)]
    
            t.set_description(f"fizz buzz (loss: {epoch_loss:.2f})")
    
    
    







    # Testeo de Red Neuronal
    # testing fiels
    # open file 
    data_file = open('mnist_dataset/mnist_test_10.csv', 'r')
    data_list = data_file.readlines()
    data_file.close()

    
    y_test, X_test = convert_to_float_vector(data_list)

    num_correct = 0
    for x, y in zip(X_test, y_test):
        predicted = argmax(feed_forward(network, x)[-1])

        actual = argmax(y)

        if predicted == actual:
            num_correct += 1

        print("predicho : {} \n actual: {}".format(predicted, actual))
        print()


    precision = int(num_correct * 100 / len(y_test))
    print("Este algoritmo tiene {}%  de precision".format(precision))

    
   





















if __name__ == "__main__": 
    main()