import numpy as np

class HammingNetwork:
    def __init__(self, prototypes, epsilon=0.1):
        self.prototypes = np.array(prototypes)
        self.num_prototypes, self.n = self.prototypes.shape

        # Весы и пороги для первого слоя (перцептрон)
        self.W = self.prototypes / 2
        self.T = np.full(self.num_prototypes, self.n / 2)

        # Параметр epsilon для сети Хопфилда
        self.epsilon = epsilon

    def run_perceptron(self, input_vector):
        # Входной вектор
        x = np.array(input_vector)

        # Считаем скалярное произведение и добавляем пороги
        y = 0.5 * (np.dot(self.W, x) + self.n)

        return y

    def run_hopfield(self, y):
        Z = np.copy(y)

        # Итерационный процесс
        while True:
            Z_new = np.copy(Z)
            for j in range(self.num_prototypes):
                S_j = Z[j] - self.epsilon * np.sum(Z) + self.epsilon * Z[j]
                Z_new[j] = S_j if S_j > 0 else 0

            if np.array_equal(Z, Z_new):
                break
            Z = Z_new

        return Z

    def predict(self, input_vector):
        y = self.run_perceptron(input_vector)
        Z = self.run_hopfield(y)
        winner_index = np.argmax(Z)
        return Z, winner_index

    def find_max_recognized_noisy_bits(self, original_pattern):
        max_flips = 0
        for num_flips in range(self.n + 1):
            noisy_pattern = original_pattern.copy()
            flip_indices = np.random.choice(self.n, num_flips, replace=False)
            noisy_pattern[flip_indices] = 1 - noisy_pattern[flip_indices]  # Инвертируем выбранные биты

            _, winner_index = self.predict(noisy_pattern)

            if np.array_equal(self.prototypes[winner_index], original_pattern):
                max_flips = num_flips
            else:
                break
        return max_flips

def print_source_vectors(patterns):
    print("Source vectors:")
    for idx, pattern in enumerate(patterns):
        print(f"y{idx+1} = {pattern.tolist()}")

def print_recall_result(idx, input_vector, normalized_vector, final_output, original_pattern):
    print(f"\nExample for y{idx+1}")
    print(f"y_original = {input_vector.tolist()}")
    print(f"winner (1) = {normalized_vector.tolist()}")
    print(f"winner (2) = {final_output.tolist()}")

    if idx == np.argmax(final_output):
        print(f"y_model (2) == y_original -> correct")
    else:
        print(f"y_model (2) != y_original -> incorrect")

if __name__ == "__main__":
    # Примеры паттернов для обучения (переведены в формат +1 и -1)
    pattern_5 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]) * 2 - 1
    pattern_6 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * 2 - 1
    pattern_9 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]) * 2 - 1

    patterns = [pattern_5, pattern_6, pattern_9]

    hamming_net = HammingNetwork(np.array(patterns))

    print_source_vectors(patterns)

    for idx, pattern in enumerate(patterns):
        input_vector = pattern.copy()
        normalized_vector = hamming_net.run_perceptron(input_vector)
        final_output, _ = hamming_net.predict(input_vector)
        print_recall_result(idx, input_vector, normalized_vector, final_output, pattern)

    for idx, pattern in enumerate(patterns):
        print(f"\nMaximum number of recognized noisy bits for y{idx+1}:")
        max_flips = hamming_net.find_max_recognized_noisy_bits(pattern)
        print(f"y{idx+1} = {max_flips}")
