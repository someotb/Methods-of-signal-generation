from matplotlib import pyplot as plt
from scipy import signal as scipy_signal
import numpy as np

fc = 10 # Несущая частота
fs = 1000 # Частота дискретизации
ns = 100 # Отсчетов на один символ
Na = 20 # Количество символов
snr = 4 # Отношение сигнал/шум

I = np.random.choice([-3, -1, 1, 3], Na)
Q = np.random.choice([-3, -1, 1, 3], Na)

signal_re = np.repeat(I, ns)
signal_im = np.repeat(Q, ns)

t = np.arange(len(signal_re)) / fs

qam_signal = signal_re * np.cos(2 * np.pi * fc * t) - signal_im * np.sin(2 * np.pi * fc * t)

qam_signal_square = qam_signal**2
avg_signal_watts = np.mean(qam_signal_square)
avg_signal_watts_db = 10 * np.log(avg_signal_watts)
avg_noise_db = avg_signal_watts_db - snr
avg_noise_watts = 10 ** (avg_noise_db / 10)
noise_volts = np.random.normal(0, np.sqrt(avg_noise_watts), len(qam_signal_square))

qam_signal_noisy = qam_signal + noise_volts

Sf = np.fft.fft(qam_signal)
freqs = np.fft.fftfreq(ns * Na, 1 / fs)
Sf_shifted = np.fft.fftshift(Sf)
freqs_shifted = np.fft.fftshift(freqs)
Pf = (np.abs(Sf_shifted) ** 2) / len(t)

Sf_noisy = np.fft.fft(qam_signal_noisy)
freqs_noisy = np.fft.fftfreq(ns * Na, 1 / fs)
Sf_shifted_noisy = np.fft.fftshift(Sf_noisy)
freqs_shifted_noisy = np.fft.fftshift(freqs_noisy)
Pf_noisy = (np.abs(Sf_shifted_noisy) ** 2) / len(t)

cos_ref = np.cos(2 * np.pi * fc * t)
sin_ref = -np.sin(2 * np.pi * fc * t)

signal_real = qam_signal_noisy * cos_ref
signal_imag = qam_signal_noisy * sin_ref

filter_len = ns
b = np.ones(filter_len) / filter_len
w, h = scipy_signal.freqz(b, worN = 8000, fs = fs)
det_I = np.convolve(signal_real, b, mode='same')
det_Q = np.convolve(signal_imag, b, mode='same')

Sf_det_I_filter = np.fft.fft(det_I)
Sf_det_Q_filter = np.fft.fft(det_Q)

Sf_filter_shift_I = np.fft.fftshift(Sf_det_I_filter)
Sf_filter_shift_Q = np.fft.fftshift(Sf_det_Q_filter)

Pf_filter_noisy_I = (np.abs(Sf_filter_shift_I) ** 2) / len(t)
Pf_filter_noisy_Q = (np.abs(Sf_filter_shift_Q) ** 2) / len(t)


plt.figure(1, figsize=(10, 4), label="Сигнал(без шума)")
plt.subplot(2, 1, 1)
plt.title("Временное представление")
plt.plot(t, qam_signal)
plt.tight_layout()
plt.grid()

plt.subplot(2, 1, 2)
plt.title("Спектральное представление")
plt.plot(freqs_shifted, Pf)
plt.tight_layout()
plt.grid()

plt.figure(2, figsize=(10, 4), label="Сигнал(с шумом)")
plt.subplot(2, 1, 1)
plt.title("Временное представление")
plt.plot(t, qam_signal_noisy)
plt.tight_layout()
plt.grid()

plt.subplot(2, 1, 2)
plt.title("Спектральное представление")
plt.plot(freqs_shifted_noisy, Pf_noisy)
plt.tight_layout()
plt.grid()

plt.figure(3, figsize=(10, 4), label="Шум")
plt.title("Нормальный шум")
plt.plot(noise_volts)
plt.tight_layout()
plt.grid()

plt.figure(4, figsize=(10, 4), label="Фильтр")
plt.subplot(2, 1, 1)
plt.title("ИХ фильтра")
plt.plot(b)
plt.tight_layout()
plt.grid()

plt.subplot(2, 1, 2)
plt.title("ЧХ фильтра")
plt.plot(w, np.abs(h))
plt.tight_layout()
plt.grid()

plt.figure(5, figsize=(10, 8), label="Сигнал после фильтра(во временной области)")
plt.subplot(2, 1, 1)
plt.plot(t, signal_re, label='Исходный I (эталон)')
plt.plot(t, det_I, label='Восстановленный I (после ФНЧ)')
plt.title("Синфазная составляющая (I)")
plt.xlabel("Время, с")
plt.legend()
plt.tight_layout()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, signal_im, label='Исходный Q (эталон)')
plt.plot(t, det_Q, label='Восстановленный Q (после ФНЧ)')
plt.title("Квадратурная составляющая (Q)")
plt.legend()
plt.xlabel("Время, с")
plt.tight_layout()
plt.grid()

plt.figure(6, figsize=(10, 8), label="Сигнал после фильтра(в частотной области)")
plt.subplot(2, 1, 1)
plt.plot(freqs_shifted, np.real(Sf_shifted), label='Исходный I (эталон)')
plt.plot(freqs_shifted_noisy, Pf_filter_noisy_I, label='Восстановленный I (после ФНЧ)')
plt.title("Синфазная составляющая (I)")
plt.xlabel("Частота, Гц")
plt.legend()
plt.tight_layout()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(freqs_shifted, np.imag(Sf_shifted), label='Исходный Q (эталон)')
plt.plot(freqs_shifted_noisy, Pf_filter_noisy_Q, label='Восстановленный Q (после ФНЧ)')
plt.title("Синфазная составляющая (Q)")
plt.xlabel("Частота, Гц")
plt.legend()
plt.tight_layout()
plt.grid()

plt.show()