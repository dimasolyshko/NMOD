import numpy as np
import matplotlib.pyplot as plt

# Hyperparams
input_size = 3
hidden_size = 70
output_size = 1
learning_rate = 0.03
epochs = 1000

class ElmanRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.Wxh = np.random.randn(hidden_size, input_size) * 0.2
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.2
        self.Why = np.random.randn(output_size, hidden_size) * 0.2
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, sequence):
        self.h_states = []  # Stores hidden states for BPTT
        self.inputs = []    # Stores inputs for BPTT
        outputs = []

        time_steps = sequence.shape[1]  # Number of time steps in the sequence
        self.reset_hidden_state()

        for t in range(time_steps):
            x_t = sequence[:, t].reshape(-1, 1)  # Extracts input for time step t
            self.inputs.append(x_t)

            # Recurrent step
            self.z = np.dot(self.Wxh, x_t) + np.dot(self.Whh, self.h) + self.bh
            self.h = np.tanh(self.z)
            self.h_states.append(self.h)

            # Output
            y_t = np.dot(self.Why, self.h) + self.by
            outputs.append(y_t.flatten())

        return np.array(outputs)

    def backward(self, y_true_sequence, y_pred_sequence):
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        # Hidden state gradient
        dh_next = np.zeros_like(self.h)

        # Back Propagation Through Time (BPTT)
        for t in reversed(range(len(y_true_sequence))):
            y_true = y_true_sequence[t].reshape(-1, 1)
            y_pred = y_pred_sequence[t].reshape(-1, 1)
            x_t = self.inputs[t]
            h_t = self.h_states[t]
            h_prev = self.h_states[t - 1] if t > 0 else np.zeros_like(h_t)

            # Output gradient
            dy = y_pred - y_true
            dWhy += np.dot(dy, h_t.T)
            dby += dy

            # Hidden state gradient
            dh = np.dot(self.Why.T, dy) + dh_next
            dz = dh * (1 - h_t ** 2)  # Derivative of tanh
            dbh += dz
            dWxh += np.dot(dz, x_t.T)
            dWhh += np.dot(dz, h_prev.T)
            dh_next = np.dot(self.Whh.T, dz)

        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby

    def reset_hidden_state(self):
        self.h = np.zeros((self.hidden_size, 1))

""" ---------------------------------------------------------------------------
Stage 1: data preparing (load data generated by generate_data.py)
"""

def load_data(file_name='data.txt'):
    x = []
    y = []
    with open(file_name, 'r') as file:
        for line in file:
            x_value, y_value = line.strip().split(',')
            x.append(float(x_value))
            y.append(float(y_value))

    x = np.array(x)
    y = np.array(y)
    return x, y

x, y = load_data('data.txt')

print("Stage 1: data preparing")
for i in range(len(x)):
    print(f"x{i+1} = {x[i]:.4f} ; y{i+1} = {y[i]:.4f}")

""" ---------------------------------------------------------------------------
Stage 2 and Stage 3 (combined):
    split data on train/test data and prepare train/test data for NN
First we create sequences, and then we divide them to train and test data
(example for sequences: y1, y2, y3 -> y5; y2, y3, y4 -> y4)
"""

def create_sliding_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

X, y_target = create_sliding_windows(y, input_size)

# Split into train and test sets (57 train, 30 test)
split_index = 120-3  # 57 training samples, 30 test samples
X_train, y_train = X[:split_index], y_target[:split_index]
X_test, y_test = X[split_index:], y_target[split_index:]

print("\nStage 2&3: split data on train/test data")
print("train_data:")
for i in range(len(X_train)):
    print(f"y{i+1} = {X_train[i][0]:.4f}, y{i+2} = {X_train[i][1]:.4f}, y{i+3} = {X_train[i][2]:.4f} -> y{i+4} = {y_train[i]:.4f}")

print(len(X_test))
print("\ntest_data:")
for i in range(len(X_test)):
    print(f"y{i+split_index+1} = {X_test[i][0]:.4f}, y{i+split_index+2} = {X_test[i][1]:.4f}, y{i+split_index+3}{X_test[i][2]:.4f} -> y{i+split_index+4} = {y_test[i]:.4f}")

""" ---------------------------------------------------------------------------
Stage 4: Train & test model
"""

print("\nStage 4: train & test model")
rnn = ElmanRNN(input_size, hidden_size, output_size, learning_rate)

train_losses = []
test_losses = []

for epoch in range(epochs):
    epoch_train_loss = 0.0
    epoch_test_loss = 0.0
    rnn.reset_hidden_state()

    # Training loop
    for i in range(len(X_train)):
        # Forward pass
        sequence = X_train[i].reshape(input_size, -1)  # Shape (input_size, time_steps)
        y_pred_seq = rnn.forward(sequence)
        y_true_seq = y_train[i].reshape(-1, 1)  # Shape (time_steps, 1)

        # Compute loss (MSE)
        train_loss = 0.5 * np.mean((y_pred_seq - y_true_seq) ** 2)
        epoch_train_loss += train_loss

        # Backward pass
        rnn.backward(y_true_seq, y_pred_seq)

    # Test loop
    for i in range(len(X_test)):
        y_pred_test_seq = rnn.forward(X_test[i].reshape(-1, 1))
        y_test_seq = y_test[i].reshape(-1, 1)
        test_loss = 0.5 * np.mean((y_pred_test_seq - y_test_seq) ** 2)
        epoch_test_loss += test_loss

    # Calculate the average loss for the epoch
    avg_train_loss = epoch_train_loss / len(X_train)
    avg_test_loss = epoch_test_loss / len(X_test)

    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)

    if epoch % 100 == 0:
        print(f"Epoch #{epoch+1}")
        print(f"train_loss: {avg_train_loss:.5f} \t test_loss: {avg_test_loss:.5f}")

    #if avg_test_loss < 1e-4:
    #    if epoch % 100 == 0:
    #        print("test_loss < 1e-4 -> stop training")
    #    break
    #else:
    #    print("test_loss > 1e-4 -> continue training")

""" ---------------------------------------------------------------------------
Stage 5: print full model outputs for best epoch
"""

print("\nStage 5: print full model outputs for best epoch")
rnn.reset_hidden_state()

predictions = []

current_input = X_test[0].copy()  # Первоначальный входной тестовый пример

for i in range(len(X_test)):
    # Выполняем прогноз на основе текущего входа
    y_pred_seq = rnn.forward(current_input.reshape(-1, 1))
    y_pred = y_pred_seq[0][0]  # Извлекаем первое значение предсказания
    predictions.append(y_pred)

    # Формируем новый входной вектор: сдвигаем и добавляем предсказанное значение
    if i + 1 < len(X_test):
        current_input = np.append(current_input[1:], y_pred)  # Удаляем первое значение, добавляем предсказанное

for i in range(len(X_test)):
    print(f"Input: {X_test[i]} -> Prediction: {predictions[i]:.4f} (Test value: {y_test[i]:.4f})")

""" ---------------------------------------------------------------------------
Draw a plot
"""

predictions = np.array(predictions)
x_test_range = x[split_index + input_size:split_index + input_size + len(X_test)]

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Actual values', color='blue', marker='o', linestyle='--')
plt.plot(x_test_range, predictions, label='Predicted values', color='red', marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()
