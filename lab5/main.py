import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

# Task 1
m = 0
var = 1
s = np.sqrt(var)
t = np.linspace(0, 10, 10000)

A1 = 0.8
A2 = 2

signal_a1 = A1 * np.sin((np.pi / 4) * t)
signal_a2 = A2 * np.sin((np.pi / 4) * t)

X1_a1 = np.random.normal(m, s, len(t)) + signal_a1
X2_a1 = np.random.uniform(-np.sqrt(3), np.sqrt(3), len(t)) + signal_a1
X1_a2 = np.random.normal(m, s, len(t)) + signal_a2
X2_a2 = np.random.uniform(-np.sqrt(3), np.sqrt(3), len(t)) + signal_a2

akf_X1_a1 = np.correlate(X1_a1, X1_a1, mode="full")
akf_X2_a1 = np.correlate(X2_a1, X2_a1, mode="full")
akf_X1_a2 = np.correlate(X1_a2, X1_a2, mode="full")
akf_X2_a2 = np.correlate(X2_a2, X2_a2, mode="full")

plt.figure(1, label="Графики сигналов s1, s2")
plt.subplot(2, 1, 1)
plt.title(f"График сигнала s1, A = {A1}")
plt.tight_layout()
plt.plot(t, signal_a1)
plt.subplot(2, 1, 2)
plt.title(f"График сигнала s2, A = {A2}")
plt.tight_layout()
plt.plot(t, signal_a2)

plt.figure(2, label=f"Графики плотности вероятности, A = {A1}")
plt.subplot(2, 1, 1)
plt.title("normal")
plt.tight_layout()
plt.hist(X1_a1, bins=50)
plt.subplot(2, 1, 2)
plt.title("uniform")
plt.tight_layout()
plt.hist(X2_a1, bins=50)

plt.figure(3, label=f"Графики плотности вероятности, A = {A2}")
plt.subplot(2, 1, 1)
plt.title("normal")
plt.tight_layout()
plt.hist(X1_a2, bins=50)
plt.subplot(2, 1, 2)
plt.title("uniform")
plt.tight_layout()
plt.hist(X2_a2, bins=50)

plt.figure(4, label="График АКФ")
plt.subplot(4, 1, 1)
plt.title(f"Нормальное распределение с A = {A1}")
plt.tight_layout()
plt.plot(akf_X1_a1)
plt.subplot(4, 1, 2)
plt.title(f"Равномерное распределение с A = {A1}")
plt.tight_layout()
plt.plot(akf_X2_a1)
plt.subplot(4, 1, 3)
plt.title(f"Нормальное распределение с A = {A2}")
plt.tight_layout()
plt.plot(akf_X1_a2)
plt.subplot(4, 1, 4)
plt.title(f"Равномерное распределение с A = {A2}")
plt.tight_layout()
plt.plot(akf_X2_a2)

# Task 2
m = 2
var = 4
s = np.sqrt(var)
t = np.linspace(0, 10, 10000)

A1 = 2

signal_a1 = A1 * np.sin((np.pi / 3) * t)

X1_a1 = np.random.normal(m, s, len(t)) + signal_a1
X2_a1 = np.random.uniform(m - 2 * np.sqrt(3), m + 2 * np.sqrt(3), len(t)) + signal_a1

akf_X1_a1 = np.correlate(X1_a1, X1_a1, mode="full")
akf_X2_a1 = np.correlate(X2_a1, X2_a1, mode="full")

plt.figure(5, label="График сигнала s1")
plt.title(f"График сигнала s1, A = {A1}")
plt.tight_layout()
plt.plot(t, signal_a1)

plt.figure(6, label=f"Графики плотности вероятности, A = {A1}")
plt.subplot(2, 1, 1)
plt.title("normal")
plt.tight_layout()
plt.hist(X1_a1, bins=50)
plt.subplot(2, 1, 2)
plt.title("uniform")
plt.tight_layout()
plt.hist(X2_a1, bins=50)

plt.figure(7, label="График АКФ")
plt.subplot(2, 1, 1)
plt.title(f"Нормальное распределение с A = {A1}")
plt.tight_layout()
plt.plot(akf_X1_a1)
plt.subplot(2, 1, 2)
plt.title(f"Равномерное распределение с A = {A1}")
plt.tight_layout()
plt.plot(akf_X2_a1)

# Task 3
m = 0
var = 4
s = np.sqrt(var)
t = np.linspace(0, 10, 10000)

white_noise = np.random.normal(m, s, len(t))

numtaps = 31
f = 0.1
ir_slow = signal.firwin(numtaps, f)
ir_high = signal.firwin(numtaps, 0.3, pass_zero=False)
w_low, hw_low = signal.freqz(ir_slow, 1)
w_high, hw_high = signal.freqz(ir_high, 1)

signal_filtered_slow = signal.lfilter(ir_slow, 1.0, white_noise)
signal_filtered_fast = signal.lfilter(ir_high, 1.0, white_noise)

akf_slow = np.correlate(signal_filtered_slow, signal_filtered_slow, mode="full")
akf_fast = np.correlate(signal_filtered_fast, signal_filtered_fast, mode="full")

plt.figure(8, label="Фильтр низких частот")
plt.title("Фильтр низких частот")
plt.subplot(2, 1, 1)
plt.xlabel("Номер отсчета ИХ")
plt.ylabel("Отсчеты ИХ фильтра")
plt.stem(ir_slow)
plt.subplot(2, 1, 2)
plt.xlabel("Номер отсчета ИХ")
plt.ylabel("Отсчеты ИХ фильтра")
plt.plot(w_low, np.abs(hw_low))
plt.xlabel("Нормированная частота")
plt.ylabel("АЧХ Фильтра")

plt.figure(9, label="Фильтр высоких частот")
plt.title("Фильтр высоких частот")
plt.subplot(2, 1, 1)
plt.xlabel("Номер отсчета ИХ")
plt.ylabel("Отсчеты ИХ фильтра")
plt.stem(ir_high)
plt.subplot(2, 1, 2)
plt.xlabel("Номер отсчета ИХ")
plt.ylabel("Отсчеты ИХ фильтра")
plt.plot(w_high, np.abs(hw_high))
plt.xlabel("Нормированная частота")
plt.ylabel("АЧХ Фильтра")

plt.figure(10, label="Графики сигналов s1_slow, s2_fast")
plt.subplot(2, 1, 1)
plt.title("График сигнала s1_slow")
plt.tight_layout()
plt.plot(t, signal_filtered_slow)
plt.subplot(2, 1, 2)
plt.title("График сигнала s2_fast")
plt.tight_layout()
plt.plot(t, signal_filtered_fast)

plt.figure(11, label="Графики плотности вероятности")
plt.subplot(2, 1, 1)
plt.title("slow")
plt.tight_layout()
plt.hist(signal_filtered_slow, bins=50)
plt.subplot(2, 1, 2)
plt.title("fast")
plt.tight_layout()
plt.hist(signal_filtered_fast, bins=50)

plt.figure(12, label="График АКФ")
plt.subplot(2, 1, 1)
plt.title("Slow")
plt.tight_layout()
plt.plot(akf_slow)
plt.subplot(2, 1, 2)
plt.title("fast")
plt.tight_layout()
plt.plot(akf_fast)

plt.show()
