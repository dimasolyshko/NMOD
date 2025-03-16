import numpy as np

class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        for pattern in patterns:
            assert len(pattern) == self.n_neurons  # Длина вектора д.б. равна кол-ву нейронов
            pattern = pattern.reshape(-1, 1)  # Преобразуем паттерн в вектор-столбец
            self.weights += pattern @ pattern.T  # Правило Хебба
        np.fill_diagonal(self.weights, 0)  # Обнуление диагональных элементов (нейрон не может иметь связи с собой)

    def recall_async(self, pattern, steps=3):
        for step in range(steps):
            for i in range(self.n_neurons):
                net_input = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if net_input >= 0 else -1
                #self.print_async_step(pattern, step+1, i)
            if np.array_equal(pattern, self.original_pattern):
                #print(" -> Relaxation, correct")
                break
        return pattern

    def recall_sync(self, pattern, steps=3):
        for step in range(steps):
            net_input = np.dot(self.weights, pattern)
            pattern = np.where(net_input >= 0, 1, -1)
            #self.print_sync_step(pattern, step+1)
            if np.array_equal(pattern, self.original_pattern):
                #print(" -> Relaxation, correct")
                break
        return pattern

    def print_async_step(self, pattern, step, neuron):
        #print(f"Stage {step}:")
        #pattern_str = [f"({pattern[i]})" if i == neuron else f"{pattern[i]}" for i in range(len(pattern))]
        #print(f"y_model ({neuron + 1}) = [{', '.join(pattern_str)}]")
        print("")

    def print_sync_step(self, pattern, step):
        #print(f"Stage {step}:")
        #print(f"y_model = [{', '.join(map(str, pattern))}]")
        print("")

    def find_max_recognized_noisy_bits(self, original_pattern, recall_func):
        # Определение максимального числа искажённых бит, которые сеть может исправить
        max_flips = 0
        for num_flips in range(self.n_neurons + 1):
            noisy_pattern = original_pattern.copy()
            flip_indices = np.random.choice(self.n_neurons, num_flips, replace=False)
            noisy_pattern[flip_indices] *= -1  # Инвертируем выбранные биты

            # print(f"\nTesting {num_flips} flipped bits for pattern {original_pattern.tolist()}")
            result_pattern = recall_func(noisy_pattern.copy())

            if np.array_equal(result_pattern, original_pattern):
                max_flips = num_flips
            else:
                break
        return max_flips

def print_source_vectors(patterns):
    print("Source vectors:")
    for idx, pattern in enumerate(patterns):
        print(f"y{idx+1} = {pattern.tolist()}")

# Функция для тестирования
if __name__ == "__main__":
    # Образы для обучения
    pattern_5 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]) * 2 - 1
    pattern_6 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * 2 - 1
    pattern_9 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]) * 2 - 1

    patterns = [pattern_5, pattern_6, pattern_9]

    # Создаем сеть Хопфилда
    hopfield_net = HopfieldNetwork(n_neurons=20)

    # Обучаем сеть на образах
    hopfield_net.train(patterns)

    # Печатаем исходные векторы
    print_source_vectors(patterns)

    # Тестирование асинхронного и синхронного восстановления для каждого образа
    for idx, pattern in enumerate(patterns):
        hopfield_net.original_pattern = pattern.copy()

        # Асинхронное восстановление
        print(f"\nAsync example for y{idx+1}")
        hopfield_net.recall_async(pattern.copy(), steps=10)

        # Синхронное восстановление
        print(f"\nSync example for y{idx+1}")
        hopfield_net.recall_sync(pattern.copy(), steps=10)

    # Определение максимального числа искажённых бит
    for idx, pattern in enumerate(patterns):
        print(f"\nMaximum number of recognized noisy bits for y{idx+1}:")

        # Для асинхронного режима
        max_flips_async = hopfield_net.find_max_recognized_noisy_bits(pattern.copy(), hopfield_net.recall_async)
        print(f"Async: y{idx+1} = {max_flips_async}")

        # Для синхронного режима
        max_flips_sync = hopfield_net.find_max_recognized_noisy_bits(pattern.copy(), hopfield_net.recall_sync)
        print(f"Sync: y{idx+1} = {max_flips_sync}")
