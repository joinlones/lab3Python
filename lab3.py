import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import Workbook

def compute_S(x, epsilon=1e-10, max_iter=1000):
    # Ряд для cos(2x)
    u_prev = 1.0  # Первый член: u_0 = 1
    sum_series = u_prev
    n = 0
    while True:
        if abs(u_prev) < epsilon or n >= max_iter:
            break
        # Рекуррентная формула для u_{n+1}
        u_next = - (4 * x**2) / ((2*n + 2)*(2*n + 1)) * u_prev
        sum_series += u_next
        u_prev = u_next
        n += 1
    return sum_series, n

# Интервал x
xn = 0  # Измените на более низкое значение для проверки
xk = 10  # Измените на более низкое значение для проверки
h = 0.5  # Уменьшите шаг для более плотной выборки

data = []
x_values = np.arange(xn, xk + h, h)

for x in x_values:
    s, n_iter = compute_S(x)
    y = np.cos(2 * x)
    data.append([x, y, s, n_iter])

df = pd.DataFrame(data, columns=['x', 'y', 's', 'n'])

# Форматирование чисел в Excel
with pd.ExcelWriter('result.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Data')
    workbook = writer.book
    worksheet = writer.sheets['Data']
    # Установка формата для столбцов B, C, D
    for column in ['B', 'C', 'D']:
        cell_range = f'{column}1:{column}{len(df)+1}'
        for row in worksheet[cell_range]:
            for cell in row:
                cell.number_format = '#,##0.000000'

print("---Результаты---")
print("Датафрейм")
print(df)

plt.figure(figsize=(8, 6))
plt.plot(df['x'], df['s'], 'g.-', label='S(x)')
plt.plot(df['x'], df['y'], 'r.-', label='Y(x)')
plt.title('График функций S(x) и Y(x)')
plt.xlabel('X')
plt.ylabel('S, Y')
plt.grid(True)
plt.legend()
plt.show()