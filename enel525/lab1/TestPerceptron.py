import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Perceptron import InputAndExpectation, Perceptron


NUMBER_INPUTS = 2
NUMBER_OUTPUTS = 1

training_inputs = [
    InputAndExpectation(np.array([[1],[4]]),np.array([[1]])),
    InputAndExpectation(np.array([[2],[5]]),np.array([[1]])),
    InputAndExpectation(np.array([[3],[3.5]]),np.array([[1]])),
    InputAndExpectation(np.array([[1],[0.5]]),np.array([[0]])),
    InputAndExpectation(np.array([[2],[2]]),np.array([[0]])),
    InputAndExpectation(np.array([[4],[0.5]]),np.array([[0]])),
]

patient = Perceptron(NUMBER_INPUTS, NUMBER_OUTPUTS)
patient.train(training_inputs)

# plot results:

for training_input in training_inputs:
    x = training_input.input_array[0][0]
    y = training_input.input_array[1][0]

    expectation = training_input.expectation_array[0]
    symbol_str = "ro" if expectation == 1 else "bo"

    plt.plot(x,y,symbol_str)

red_patch = mpatches.Patch(color='red', label='One, AKA class 1')
blue_patch = mpatches.Patch(color='blue', label='Zero, AKA class 2')

# Plot decision boundary
weight_array = patient.weight_matrix
bias = patient.bias_vector[0][0]

slope = (-1) * weight_array[0][0] / weight_array[0][1]
y_intercept = (-1) * bias / weight_array[0][1]

x_vals = np.array(plt.gca().get_xlim())
y_vals = y_intercept + slope * x_vals
plt.plot(x_vals, y_vals, '--')


plt.legend(handles=[red_patch, blue_patch])
plt.show()


# Make sure predictions are correct for training inputs
test_results = []
for training_input in training_inputs:
    prediction = patient.predict(training_input.input_array)
    test_results.append(prediction == training_input.expectation_array)

print(test_results)
