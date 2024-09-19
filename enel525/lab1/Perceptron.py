import numpy as np


class InputAndExpectation:
    def __init__(self, input_array, expectation_array):
        self.input_array = input_array
        self.expectation_array = expectation_array


class Perceptron:
    def __init__(self, number_inputs, number_outputs):
        self.number_inputs = number_inputs
        self.number_outputs = number_outputs
        self.weight_matrix = np.zeros((self.number_outputs, self.number_inputs))
        self.bias_vector = np.zeros((self.number_outputs, 1))

    def train(self, training_point_list):
        # flag represents whether or not the current decision boundary accounts for this point
        training_point_to_flag_dict = {}

        for training_point in training_point_list:
            if self.__training_point_has_valid_dimensions(training_point):
                training_point_to_flag_dict.update({training_point: False})
            else:
                print("Found input with incorrect dimensions, exitting")
                return

        while True:
            # Start by unsetting all the flags.  Everything will need to be rechecked
            for training_point in training_point_to_flag_dict.keys():
                training_point_to_flag_dict[training_point] = False

            for training_point in training_point_to_flag_dict.keys():
                prediction = self.predict(training_point.input_array)
                error = training_point.expectation_array - prediction
                if error == np.zeros((self.number_outputs, 1)):
                    training_point_to_flag_dict[training_point] = True
                else:
                    new_weight_matrix = self.weight_matrix + np.matmul(
                        error, np.transpose(training_point.input_array)
                    )
                    new_bias_vector = self.bias_vector + error
                    self.weight_matrix = new_weight_matrix
                    self.bias_vector = new_bias_vector
            if all(training_point_to_flag_dict.values()):
                # break when all inputs accounted for
                break

    def predict(self, input_array):
        n = np.matmul(self.weight_matrix, input_array) + self.bias_vector
        hard_limiter = lambda x: 1 if x >= 0 else 0
        return hard_limiter(n)

    def __training_point_has_valid_dimensions(self, training_point):
        return self.__verify_vector_size(
            training_point.input_array, self.number_inputs
        ) and self.__verify_vector_size(
            training_point.expectation_array, self.number_outputs
        )

    def __verify_vector_size(self, vector, expected_rows):
        return vector.shape == (expected_rows, 1)
