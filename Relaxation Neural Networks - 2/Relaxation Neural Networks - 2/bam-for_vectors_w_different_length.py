import numpy as np

class BAM:  # Bidirectional Associative Memory
    def __init__(self, input_size, output_size):
        # Initialize weight matrix
        self.weights = np.zeros((input_size, output_size))

    def train(self, input_vector, output_vector):
        # Update weights using Hebbian learning rule
        self.weights += np.outer(input_vector, output_vector)

    def recall(self, input_vector=None, output_vector=None):
        if input_vector is not None:
            # Recall output vector from input
            return np.sign(np.dot(self.weights.T, input_vector))
        elif output_vector is not None:
            # Recall input vector from output
            return np.sign(np.dot(self.weights, output_vector))

    def relaxation(self, input_vector=None, output_vector=None):
        # Update input and output vectors
        if input_vector is not None:
            y_model = self.recall(input_vector=input_vector)
            x_model = self.recall(output_vector=y_model)
            return x_model, y_model
        elif output_vector is not None:
            x_model = self.recall(output_vector=output_vector)
            y_model = self.recall(input_vector=x_model)
            return x_model, y_model

    def find_max_recognized_noisy_bits(self, original_pattern, is_input=True):
        size = len(original_pattern)
        max_flips = 0
        for num_flips in range(size + 1):
            noisy_pattern = original_pattern.copy()
            flip_indices = np.random.choice(size, num_flips, replace=False)
            noisy_pattern[flip_indices] *= -1  # Flip selected bits

            if is_input:
                # Handle noisy input pattern
                _, y_model = self.relaxation(input_vector=noisy_pattern)
                result_pattern = self.recall(output_vector=y_model)
            else:
                # Handle noisy output pattern
                _, x_model = self.relaxation(output_vector=noisy_pattern)
                result_pattern = self.recall(output_vector=x_model)

            if np.array_equal(result_pattern, original_pattern):
                max_flips = num_flips
            else:
                break
        return max_flips


def print_source_vectors(input_patterns, output_patterns):
    print("Source vectors:")
    for idx, (x, y) in enumerate(zip(input_patterns, output_patterns)):
        print(f"x{idx+1} = {x.tolist()}; y{idx+1} = {y.tolist()}")

def print_recall_result(idx, x_original, y_original, x_model, y_model):
    print(f"\nExample for x{idx+1}")
    print(f"x_original = {x_original.tolist()}")
    print("Stage 1:")
    print(f"y_model (1) = {y_model.tolist()}")
    print(f"x_model (1) = {x_model.tolist()}")
    if np.array_equal(x_model, x_original):
        print("x_model (1) == x_original -> relaxation, correct")
    else:
        print("x_model (1) != x_original -> incorrect")

def print_recall_result_y(idx, y_original, x_model, y_model):
    print(f"\nExample for y{idx+1}")
    print(f"y_original = {y_original.tolist()}")
    print("Stage 1:")
    print(f"x_model (1) = {x_model.tolist()}")
    print(f"y_model (1) = {y_model.tolist()}")
    if np.array_equal(y_model, y_original):
        print("y_model (1) == y_original -> relaxation, correct")
    else:
        print("y_model (1) != y_original -> incorrect")

if __name__ == "__main__":
    # Input and output vectors
    pattern_1_in = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]) * 2 - 1
    pattern_1_out = np.array([1, 0, 1, 0]) * 2 - 1

    pattern_2_in = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]) * 2 - 1
    pattern_2_out = np.array([1, 1, 0, 0]) * 2 - 1

    input_patterns = [pattern_1_in, pattern_2_in]
    output_patterns = [pattern_1_out, pattern_2_out]

    bam = BAM(input_size=11, output_size=4)

    # Train the network
    for x, y in zip(input_patterns, output_patterns):
        bam.train(x, y)

    print_source_vectors(input_patterns, output_patterns)

    # Test recall for each input vector
    for idx, (x_original, y_original) in enumerate(zip(input_patterns, output_patterns)):
        x_model, y_model = bam.relaxation(input_vector=x_original)
        print_recall_result(idx, x_original, y_original, x_model, y_model)

    # Test recall for each output vector
    for idx, (x_original, y_original) in enumerate(zip(input_patterns, output_patterns)):
        x_model, y_model = bam.relaxation(output_vector=y_original)
        print_recall_result_y(idx, y_original, x_model, y_model)

    # Find max recognized noisy bits
    for idx, pattern in enumerate(input_patterns):
        print(f"\nMaximum number of recognized noisy bits for x{idx+1}:")
        max_flips = bam.find_max_recognized_noisy_bits(pattern, is_input=True)
        print(f"x{idx+1} = {max_flips}")

    for idx, pattern in enumerate(output_patterns):
        print(f"\nMaximum number of recognized noisy bits for y{idx+1}:")
        max_flips = bam.find_max_recognized_noisy_bits(pattern, is_input=False)
        print(f"y{idx+1} = {max_flips}")
