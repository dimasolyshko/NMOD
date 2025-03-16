import numpy as np

def load_data(file_name='data.txt'):
    X = []
    y = []
    with open(file_name, 'r') as file:
        for line in file:
            x_value, y_value = line.strip().split(',')
            X.append(float(x_value))
            y.append(float(y_value))

    X = np.array(X)
    y = np.array(y)
    return X, y

# Подготовка обучающих данных
def create_sequences(y, seq_length):
    inputs, outputs = [], []
    for i in range(len(y) - seq_length):
        inputs.append(y[i:i+seq_length])
        outputs.append(y[i+seq_length])
    return np.array(inputs), np.array(outputs)

# Линейная нейронная сеть с одним слоем
class LinearNN:
    def __init__(self, input_size, lr):
        # Инициализация весов и порога(?) случайным образом
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.lr = lr

    def predict(self, inputs):
        return np.dot(inputs, self.weights) + self.bias #y_1=∑_(i=1)^n〖w_i1 x_i-T_1 〗

    def train(self, inputs, outputs, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(outputs)):
                outputs_pred = self.predict(inputs[i])
                error = outputs_pred - outputs[i]
                total_loss += error**2/2
                # Градиентный спуск
                self.weights -= self.lr * error * inputs[i]
                self.bias -= self.lr * error
            avg_loss = total_loss / len(outputs)
            print(f'Epoch #{epoch+1}, Loss: {avg_loss:.5f}')
            if avg_loss < 1e-4:
                print('-> stop training')
                break
            print('-> continue training')
        return avg_loss

def main():
    # Параметры
    num_train = 30
    num_test = 15
    sequence_length = 3
    epochs = 100
    learning_rate = 0.09

    # Этап 1: Подготовка данных
    print("Stage 1: data preparing")
    x, y = load_data('D:/Neironki/Predictive Linear NN/Predictive Linear NN/data.txt')
    for i in range(len(x)):
        print(f"x{i+1} = {x[i]:.5f}; y{i+1} = {y[i]:.5f}")

    # Этап 2: Разделение на обучающие и тестовые данные
    print("\nStage 2: split data on train/test data")
    x_train, y_train = x[:num_train], y[:num_train]
    x_test, y_test = x[num_train:], y[num_train:]

    print("train_data:")
    for i in range(len(x_train)):
        print(f"x{i+1} = {x_train[i]:.5f}; y{i+1} = {y_train[i]:.5f}")

    print("\ntest_data:")
    for i in range(len(x_test)):
        print(f"x{i+num_train+1} = {x_test[i]:.5f}; y{i+1} = {y_test[i]:.5f}")

    # Этап 3: Подготовка данных для обучения
    print("\nStage 3: prepare train/test data for NN")
    X_train, Y_train = create_sequences(y_train, sequence_length)
    X_test, Y_test = create_sequences(y_test, sequence_length)

    # Вывод примера последовательностей
    print("\ntrain_data:")
    for i in range(len(X_train)):
        print(f"{X_train[i]} -> {Y_train[i]:.5f}")

    print("\ntest_data:")
    #for i in range(len(X_test)):
    print(f"{X_test[0]} -> (original: {Y_test[0]:.5f})")

    # Этап 4: Обучение и тестирование модели
    print("\nStage 4: train & test model")
    model = LinearNN(input_size=sequence_length, lr=learning_rate)
    train_loss = model.train(X_train, Y_train, epochs)

    # Тестирование модели
    predictions = []
    total_test_loss = 0
    for i in range(len(X_test)):
        pred = model.predict(X_test[i])
        predictions.append(pred)
        error = pred - Y_test[i]
        total_test_loss += error**2/2
    test_loss = total_test_loss / len(Y_test)
    print(f"\nFinal test loss: {test_loss:.5f}")

    # Этап 5: Вывод прогноза модели
    print("\nStage 5: print full model outputs for best epoch")
    for i in range(len(X_test)):
        print(f"{X_test[i]} -> {predictions[i]:.5f} (original: {Y_test[i]:.5f})")

if __name__ == "__main__":
    main()
