import random
from cProfile import label

import numpy as np
from matplotlib import pyplot as plt


def qpsk_mapper(bits: list):
    symbols = np.zeros(len(bits) // 2, dtype=complex)
    for i in range(len(bits) // 2):
        symbols[i] = complex(
            1.0 - 2.0 * bits[2 * i + 0], 1.0 - 2.0 * bits[2 * i + 1]
        ) / np.sqrt(2.0)
    return symbols


def upsampler(symbols, upsample_koef):
    up_symbols = np.zeros(len(symbols) * upsample_koef, dtype=complex)
    for i in range(len(symbols)):
        up_symbols[i * upsample_koef] = symbols[i]
    return up_symbols


def downsample(symbols, upsample_koef):
    return symbols[::upsample_koef]


def white_noise(conv_symbols, snr_db):
    cos2 = np.abs(conv_symbols) ** 2
    sig_awg_watts = np.mean(cos2)
    sig_awg_db = 10 * np.log10(sig_awg_watts)
    noise_awg_db = sig_awg_db - snr_db
    noise_awg_watts = 10 ** (noise_awg_db / 10)
    mean_noise = 0

    noise_volts_re = np.random.normal(
        mean_noise, np.sqrt(noise_awg_watts), len(conv_symbols)
    )
    noise_volts_im = np.random.normal(
        mean_noise, np.sqrt(noise_awg_watts), len(conv_symbols)
    )
    n_compl = noise_volts_re + 1j * noise_volts_im
    cs_noisy = conv_symbols + n_compl
    return cs_noisy


def demapper(symbols):
    demapped_bits = np.zeros(len(symbols) * 2, dtype=int)
    kor = np.sqrt(2.0)
    znach = {
        ((1 + 1j) / kor): (0, 0),
        ((1 - 1j) / kor): (0, 1),
        ((-1 - 1j) / kor): (1, 1),
        ((-1 + 1j) / kor): (1, 0),
    }

    for i in range(len(symbols)):
        min_ras = float("inf")
        best_bits = (0, 0)
        for k, z in znach.items():
            rast = np.sqrt(
                (symbols[i].real - k.real) ** 2 + (symbols[i].imag - k.imag) ** 2
            )
            if rast < min_ras:
                min_ras = rast
                best_bits = z

        demapped_bits[2 * i] = best_bits[0]
        demapped_bits[2 * i + 1] = best_bits[1]
    return demapped_bits


bits = []
up_koef = 10
snr_db = 3
ih = np.ones(up_koef)

for i in range(1000):
    bits.append(random.randint(0, 1))

# TX
symbols = qpsk_mapper(bits)
up_symbols = upsampler(symbols, up_koef)
conv_symbols = np.convolve(up_symbols, ih, mode="same")
symbols_n = white_noise(conv_symbols, snr_db)

# RX
rx_symbols_conv = np.convolve(symbols_n, ih, mode="same") / up_koef
rx_down_symbols = downsample(rx_symbols_conv, up_koef)
dem_symb = demapper(rx_down_symbols)

plt.figure(1, label="Bits")
plt.plot(bits)

plt.figure(2, label="Symbols")
plt.subplot(2, 1, 1)
plt.plot(np.real(symbols))
plt.plot(np.imag(symbols))
plt.subplot(2, 1, 2)
plt.scatter(np.real(symbols), np.imag(symbols))


plt.figure(3, label="UpSample symbols")
plt.subplot(2, 1, 1)
plt.plot(np.real(up_symbols))
plt.plot(np.imag(up_symbols))
plt.subplot(2, 1, 2)
plt.scatter(np.real(up_symbols), np.imag(up_symbols))


plt.figure(4, label="Symbols after Convolve")
plt.subplot(2, 1, 1)
plt.plot(np.real(conv_symbols))
plt.plot(np.imag(conv_symbols))
plt.subplot(2, 1, 2)
plt.scatter(np.real(conv_symbols), np.imag(conv_symbols))

plt.figure(5, label="Noisy Symbols")
plt.subplot(2, 1, 1)
plt.plot(np.real(symbols_n))
plt.plot(np.imag(symbols_n))
plt.subplot(2, 1, 2)
plt.scatter(np.real(symbols_n), np.imag(symbols_n))

plt.figure(6, label="RX symbols")
plt.subplot(2, 1, 1)
plt.plot(np.real(rx_symbols_conv))
plt.plot(np.imag(rx_symbols_conv))
plt.subplot(2, 1, 2)
plt.scatter(np.real(rx_symbols_conv), np.imag(rx_symbols_conv))

plt.figure(7, label="RX Symbols after down sampling")
plt.subplot(2, 1, 1)
plt.plot(np.real(rx_down_symbols))
plt.plot(np.imag(rx_down_symbols))
plt.subplot(2, 1, 2)
plt.scatter(np.real(rx_down_symbols), np.imag(rx_down_symbols))

plt.figure(8, label="Decoded Bits vs Bits")
plt.plot(dem_symb, label="Dec bits")
plt.plot(bits, label="bits")
plt.legend()

plt.show()
