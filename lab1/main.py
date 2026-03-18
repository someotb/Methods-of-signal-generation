from matplotlib import pyplot as plt
import numpy as np
import cmath
import math

# Вычисление ПФ прямоугольного сигнала на основе численного интегрирования
tau = 0.003
T = 0.01
fs = 44100
dt = 1 / fs
signal = []
ampl_1 = []
argu_1 = []
freques = range(0, 1001, 5)
N = int(T / dt)

for i in range(N):
    t = i * dt
    if t < tau:
        signal.append(1)
    else:
        signal.append(0)

for f in freques:
    fft = 0 + 0j
    for i in range(len(signal)):
        t = i * dt
        exponenta = cmath.exp(-1j * 2 * math.pi * f * t)
        fft += signal[i] * exponenta * dt
    ampl_1.append(abs(fft))
    argu_1.append(cmath.phase(fft))

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(freques, ampl_1)
plt.subplot(2, 1, 2)
plt.plot(freques, argu_1)

# Проверка свойства ПФ сигнала (свойство модуляции, смещения спектра)
f_carrier = 1e3
freques = np.linspace(0, 2000, 2001)
ampl_2 = []
argu_2 = []

t = np.arange(0, T, dt)
rect = np.where(t < tau, 1.0, 0.0)
carrier = np.cos(2 * np.pi * f_carrier * t)
radio_signal = rect * carrier

for f in freques:
    integrant = radio_signal * np.exp(-1j * 2 * np.pi * f * t)
    fft = np.sum(integrant) * dt
    ampl_2.append(np.abs(fft))
    argu_2.append(np.angle(fft))

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(freques, ampl_2)
plt.subplot(2, 1, 2)
plt.plot(freques, argu_2)

# Спектр последовательности, вычисление при помощи ДПФ
T = 0.01 # Длительность сигнала
ns = 64 # Oтсчетов на символ
Na = 128
Ns = ns * Na
fs = ns / T
Ts = 1 / fs

bits = np.random.randint(0, 2, Na)
signal = np.repeat(bits, ns)

Sf = np.fft.fft(signal)
freqs_axis = np.fft.fftfreq(Ns, Ts)

Sf_shifted = np.fft.fftshift(Sf)
freqs_shifted = np.fft.fftshift(freqs_axis)

Pf = (Sf_shifted * np.conj(Sf_shifted)).real * Ts / Ns

plt.figure(3)
plt.plot(freqs_shifted, Pf)
plt.xlim(-750, 750)

plt.show()