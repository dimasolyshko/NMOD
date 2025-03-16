import numpy as np
import matplotlib.pyplot as plt
from albumentations import Compose, GaussianBlur, GaussNoise
import time
import os
import copy

""" ---------------------------------------------------------------------------
Defining the Multilayer Neural Network: parameters and architecture
"""

# Hyperparams
learning_rate = 0.04
fine_tuning_learning_rate = 0.001  # Снижает скорость обучения для fine-tuning

batch_size = 64

input_size = 784  # 28x28 изображения
hidden_size = 800
output_size = 10

max_epochs = 100 # Не хотите убиться при первом запуске - поставьте штук 201
fine_tuning_epochs = 5  # Дополнительные эпохи для fine-tuning


print(f"-----------------------------\n"
      f"NN architecture + hyperparams\n"
      f"-----------------------------\n")

print(f"Architecture: {input_size} - {hidden_size} (Leaky ReLU) - {output_size}")
print(f"Learning rate: {learning_rate}")
print(f"Batch size: {batch_size}")
print(f"Max train epoch: {max_epochs}")
print(f"Fine-tuning learning rate: {fine_tuning_learning_rate}")
print(f"Fine-tuning epochs: {fine_tuning_epochs}")

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros(hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros(output_size)

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, x):
        # Single hidden layer
        self.z1 = x.dot(self.weights1) + self.bias1
        self.a1 = self.leaky_relu(self.z1)
        # Output layer
        self.z2 = self.a1.dot(self.weights2) + self.bias2
        return self.softmax(self.z2)

    def backward(self, x, y, output, lr):
        y_true = np.zeros((y.size, output.shape[1]))
        y_true[np.arange(y.size), y] = 1

        # For output layer
        d_z2 = (output - y_true) / y_true.shape[0]
        d_weights2 = self.a1.T.dot(d_z2)
        d_bias2 = np.sum(d_z2, axis=0)

        # For hidden layer
        d_a1 = d_z2.dot(self.weights2.T)
        d_z1 = d_a1 * self.leaky_relu_derivative(self.z1)
        d_weights1 = x.T.dot(d_z1)
        d_bias1 = np.sum(d_z1, axis=0)

        # Update weights and biases
        self.weights1 -= lr * d_weights1
        self.bias1 -= lr * d_bias1
        self.weights2 -= lr * d_weights2
        self.bias2 -= lr * d_bias2

    def train_batch(self, X_batch, y_batch, lr):
        output = self.forward(X_batch)
        self.backward(X_batch, y_batch, output, lr)
        predictions = np.argmax(output, axis=1)
        return np.mean(predictions == y_batch)

    def evaluate(self, X, y):
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        return np.mean(predictions == y)

    def save_weights(self, filepath_prefix):
        """Сохраняет веса и пороги модели в файлы."""
        np.save(f"{filepath_prefix}_weights1.npy", self.weights1)
        np.save(f"{filepath_prefix}_bias1.npy", self.bias1)
        np.save(f"{filepath_prefix}_weights2.npy", self.weights2)
        np.save(f"{filepath_prefix}_bias2.npy", self.bias2)
        print(f"Модель сохранена в {filepath_prefix}_weights1.npy и других файлах.")

    def load_weights(self, filepath_prefix):
        """Загружает веса и пороги модели из файлов."""
        self.weights1 = np.load(f"{filepath_prefix}_weights1.npy")
        self.bias1 = np.load(f"{filepath_prefix}_bias1.npy")
        self.weights2 = np.load(f"{filepath_prefix}_weights2.npy")
        self.bias2 = np.load(f"{filepath_prefix}_bias2.npy")
        print(f"Модель загружена из {filepath_prefix}_weights1.npy и других файлов.")

""" ---------------------------------------------------------------------------
Acquire images from MNIST dataset and apply transformations to them
    (BY VARIANT, HERE: NOISE + BLUR)
"""

# Путь к данным
base_dir = "C:/Users/Disoland/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1"
train_images_path = os.path.join(base_dir, "train-images.idx3-ubyte")
train_labels_path = os.path.join(base_dir, "train-labels.idx1-ubyte")
test_images_path = os.path.join(base_dir, "t10k-images.idx3-ubyte")
test_labels_path = os.path.join(base_dir, "t10k-labels.idx1-ubyte")

# Загрузка и подготовка данных MNIST
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        f.read(16)
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
    return images / 255.0

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Загрузка данных в массивы
X_train = load_mnist_images(train_images_path)
y_train = load_mnist_labels(train_labels_path)
X_test = load_mnist_images(test_images_path)
y_test = load_mnist_labels(test_labels_path)

# Преобразования данных с использованием albumentations
#(ПО ВАРИАНТУ, p - это вероятность срабатывания, выставить 1.0, чтобы срабатывало всегда)
transform = Compose([
    GaussianBlur(blur_limit=(3, 7), p=0.9),
    GaussNoise(var_limit=(800.0, 1000.0), p=0.9),
])

def apply_transformations(image):
    """Функция для применения преобразований к изображению."""
    image = (image * 255).astype(np.uint8)  # Преобразуем к uint8 для albumentations
    image = image.reshape(28, 28, 1)  # Форматируем в 28x28 для albumentations
    augmented = transform(image=image)  # Применение трансформации
    transformed_image = augmented["image"].astype(np.float32) / 255.0  # Приведение обратно к float
    return transformed_image.flatten()  # Преобразуем обратно к вектору

# Накладываем фильтры на изображения
X_train_augmented = np.array([apply_transformations(x) for x in X_train])

""" ---------------------------------------------------------------------------
Print all console output + train and test the network
    FOR THE NN TO START TRAINING, YOU NEED TO CLOSE THE WINDOW WITH THE IMAGE
"""

# Ввод пользователя для выбора действия
choice = input("\nВыберите действие:\n1) Тренировать нейронную сеть\n2) Протестировать нейронную сеть\nВведите 1 или 2: ").strip()

# Визуализация примеров преобразования
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
original_image = X_train[0].reshape(28, 28)
transformed_image = apply_transformations(X_train[0]).reshape(28, 28)

axes[0].imshow(original_image, cmap="gray")
axes[0].set_title("Оригинальное изображение")
axes[1].imshow(transformed_image, cmap="gray")
axes[1].set_title("После предобработки")
plt.show()
# Конец отрывка, выводит пример оригинального изображения и изображения с наложенными фильтрами
# Если фильтры вдруг не накладываются, проверьте параметр p в trasnform (начало на 142 строке)
# Примечание: нейронка будет работать лучше, если ~0.2 < p < ~0.5

if choice == '1':
    # Обучение нейронной сети
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    train_accuracies, test_accuracies = [], []
    best_test_accuracy = 0
    best_model_state = None

    print(f"\n-----------------------------\n"
          f"Train & Test process\n"
          f"-----------------------------\n")

    for epoch in range(1, max_epochs + 1):
        start_time = time.time()
        train_accuracy_epoch = []

        for i in range(0, X_train_augmented.shape[0], batch_size):
            X_batch = X_train_augmented[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            train_accuracy_epoch.append(nn.train_batch(X_batch, y_batch, learning_rate))

        train_accuracy = np.mean(train_accuracy_epoch)
        test_accuracy = nn.evaluate(X_test.reshape(-1, 784), y_test)
        epoch_time = time.time() - start_time

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_model_state = copy.deepcopy(nn)

        print(f"Epoch #{epoch}: Train accuracy = {train_accuracy:.4f}, "
              f"Test accuracy = {test_accuracy:.4f}, Time = {epoch_time:.2f}s")

        if test_accuracy > 0.98:
            print(f"Достигнута требуемая точность на эпохе #{epoch}")
            break

    # Fine-tuning
    if best_test_accuracy < 0.98:
        print("\nFine-tuning...")
        nn = best_model_state  # Восстанавливаем лучшие веса
        for fine_epoch in range(1, fine_tuning_epochs + 1):
            start_time = time.time()
            train_accuracy_epoch = []

            for i in range(0, X_train_augmented.shape[0], batch_size):
                X_batch = X_train_augmented[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                train_accuracy_epoch.append(nn.train_batch(X_batch, y_batch, fine_tuning_learning_rate))

            train_accuracy = np.mean(train_accuracy_epoch)
            test_accuracy = nn.evaluate(X_test.reshape(-1, 784), y_test)
            epoch_time = time.time() - start_time

            print(f"Fine-tuning Epoch #{fine_epoch}: Train accuracy = {train_accuracy:.4f}, "
                  f"Test accuracy = {test_accuracy:.4f}, Time = {epoch_time:.2f}s")

            if test_accuracy > 0.98:
                print(f"Точность 0.98 достигнута во время fine-tuning на эпохе #{fine_epoch}")
                break

    # Воспроизведение звука в конце обучения
    os.system('echo "\a"')

    # Запрос на сохранение модели
    save_model = input("\nХотите сохранить веса и пороги? (y/n): ").strip().lower()
    if save_model == 'y':
        # Сохраняем веса и пороги модели
        model_save_path = "best_model"  # Путь для сохранения модели
        nn.save_weights(model_save_path)
        print(f"Модель успешно сохранена в {model_save_path}_weights1.npy и др.")
elif choice == '2':
    # Тестирование нейронной сети
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    model_path = input("Введите путь к сохранённой модели (например, best_model): ").strip()

    nn.load_weights(model_path)

    test_accuracy = nn.evaluate(X_test.reshape(-1, 784), y_test)
    print(f"Точность на тестовых данных: {test_accuracy:.4f}")
else:
    print("Неверный выбор. Завершаю выполнение программы.")
