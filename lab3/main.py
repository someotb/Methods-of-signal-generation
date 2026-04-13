import numpy as np
from matplotlib import pyplot as plt


def get_pdf(mean, var, x_array):
    return np.exp(-((x_array - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


def get_empirical_pdf(xn, bins, step):
    counts = np.zeros(len(bins) - 1)

    for value in xn:
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1]:
                counts[i] += 1
                break

    empirical_pdf = counts / (len(xn) * step)
    return empirical_pdf


# Task 1
mean_0 = 0
mean_m1 = -1
var_1 = 1
var_3 = 3
var_02 = 0.2
w_0_1 = []
w_0_3 = []
w_0_02 = []
w_m1_1 = []
x_array = np.arange(-5, 5, 0.01)

w_0_1 = get_pdf(mean_0, var_1, x_array)
w_0_3 = get_pdf(mean_0, var_3, x_array)
w_0_02 = get_pdf(mean_0, var_02, x_array)
w_m1_1 = get_pdf(mean_m1, var_1, x_array)

plt.figure(1, label="График плотности распределения")
plt.title("График плотности распределения")
plt.plot(x_array, w_0_1, color="r", label=f"mx = {mean_0}, var = {var_1}")
plt.plot(x_array, w_0_3, color="b", label=f"mx = {mean_0}, var = {var_3}")
plt.plot(x_array, w_0_02, color="y", label=f"mx = {mean_0}, var = {var_02}")
plt.plot(x_array, w_m1_1, color="g", label=f"mx = {mean_m1}, var = {var_1}")
plt.legend()

# Task 2
t = np.linspace(0, 3, 1000)
s1 = np.sqrt(var_1)
s3 = np.sqrt(var_3)
s02 = np.sqrt(var_02)

xn_1 = np.random.normal(mean_0, s1, len(t))
xn_3 = np.random.normal(mean_0, s3, len(t))
xn_02 = np.random.normal(mean_0, s02, len(t))

plt.figure(2, label="Вектор значений СВ с нормальным распределением")
plt.title("Вектор значений СВ")
plt.scatter(t, xn_1, s=1, color="r", label=f"mx = {mean_0}, s = {s1}")
plt.scatter(t, xn_3, s=1, color="b", label=f"mx = {mean_0}, s = {s3}")
plt.scatter(t, xn_02, s=1, color="y", label=f"mx = {mean_0}, s = {s02}")
plt.legend()

# Task 3
mx = 0
var = 3
s = np.sqrt(var)

data = np.random.normal(loc=mx, scale=s, size=10000)

step = 0.2  # Ширина одного ящика(bin)
bins_edges = np.arange(-6, 6 + step, step)
bins_centers = bins_edges[:-1] + step / 2
bins_counters = np.zeros_like(bins_centers)

for x in data:
    for e in range(len(bins_edges) - 1):
        if bins_edges[e] <= x < bins_edges[e + 1]:
            bins_counters[e] += 1
            break

norm_var = len(data) * step
bins_counters /= norm_var

plt.figure(3, label="Гистаграмма распрелеления")
plt.title(f"Гистаграмма распрелеления[mx = {mx}, var = {var}]")
plt.plot(bins_centers, bins_counters)

# Task 4
emp_mx = np.sum(data) / len(data)
emp_var = np.sum(data**2) / len(data) - emp_mx**2
print("По исходным данным")
print(f"Set mx: {mx} | Get mx: {emp_mx}")
print(f"Set var: {var} | Get var: {emp_var}")

# Task 5
m_x = np.sum(bins_centers * bins_counters * step)
va_r = np.sum((bins_centers**2) * bins_counters * step) - m_x**2

print("По гистаграмме распрелеления")
print(f"Set mx: {mx} | Get mx: {m_x}")
print(f"Set var: {var} | Get var: {va_r}")

plt.show()
