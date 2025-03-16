import numpy as np
import matplotlib.pyplot as plt

# Функция f(x) = sin(5x) * cos(4x) + cos(x)
def f(x):
    return np.sin(5 * x) * np.cos(4 * x) + np.cos(x)

# Генерация данных
def generate_data(num_points=180):
    x = np.linspace(0, 4 * np.pi, num_points)  # Равномерно распределенные точки от 0 до 10
    y = f(x)  # Вычисление значений функции для каждой точки x

    # Сохранение данных в файл
    with open('data.txt', 'w') as file:
        for i in range(num_points):
            file.write(f"{x[i]},{y[i]}\n")

    return x, y

# Генерация данных
x, y = generate_data()

# Построение графика
plt.plot(x, y, label="Data")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Generated Data")
plt.legend()
plt.show()
