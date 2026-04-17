from matplotlib import pyplot as plt
import numpy as np
import random

def qpsk_mapper(bits: list):
    symbols = np.zeros(len(bits) // 2, dtype=complex)
    for i in range(len(bits) // 2):
        symbols[i] = (complex(1.0 - 2.0 * bits[2 * i + 0], 1.0 - 2.0 * bits[2 * i + 1]) / np.sqrt(2.0)) 
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


    noise_volts_re = np.random.normal(mean_noise, np.sqrt(noise_awg_watts), len(conv_symbols))
    noise_volts_im = np.random.normal(mean_noise, np.sqrt(noise_awg_watts), len(conv_symbols))
    n_compl = noise_volts_re + 1j * noise_volts_im
    cs_noisy = conv_symbols + n_compl
    return cs_noisy

def demapper(symbols):
    demapped_bits = np.zeros(len(symbols), dtype=list)
    kor = np.sqrt(2.0)
    znach = {(1 + 1j / kor): (0, 0),
             (1 - 1j / kor): (0, 1),
             (-1 - 1j / kor): (1, 1),
             (-1 + 1j / kor): (1, 0)}
    
    for i in range(len(symbols) // 2):
        min_ras = 10
        symb = 0 + 1j
        for k, z in znach.items():
            rast = np.sqrt((symbols[i].real - k.real)**2 + (symbols[i].imag - k.imag)**2)
            if rast < min_ras:
                min_ras = rast
                symb = z
        demapped_bits[2 * i] = symb[0]
        demapped_bits[2 * i + 1] = symb[1]
    return demapped_bits

        
bits = []
up_koef = 10
snr_db = 10
ih = np.ones(up_koef)

for i in range(1000):
    bits.append(random.randint(0, 1))

# TX
symbols = qpsk_mapper(bits)
up_symbols = upsampler(symbols, up_koef)
conv_symbols = np.convolve(up_symbols, ih)
symbols_n = white_noise(conv_symbols, snr_db)

# RX
rx_symbols_conv = np.convolve(symbols_n, ih) / up_koef
rx_down_symbols = downsample(rx_symbols_conv, up_koef)
dem_symb = demapper(rx_down_symbols)

plt.figure(1)
plt.plot(bits)

plt.figure(2)
plt.subplot(2,1,1)
plt.plot(np.real(symbols))
plt.plot(np.imag(symbols))
plt.subplot(2,1,2)
plt.scatter(symbols.real, symbols.imag)


plt.figure(3)
plt.subplot(2,1,1)
plt.plot(np.real(up_symbols))
plt.plot(np.imag(up_symbols))
plt.subplot(2,1,2)
plt.scatter(up_symbols.real, up_symbols.imag)


plt.figure(4)
plt.subplot(2,1,1)
plt.plot(np.real(conv_symbols))
plt.plot(np.imag(conv_symbols))
plt.subplot(2,1,2)
plt.scatter(conv_symbols.real, conv_symbols.imag)

plt.figure(5)
plt.subplot(2,1,1)
plt.plot(np.real(symbols_n))
plt.plot(np.imag(symbols_n))
plt.subplot(2,1,2)
plt.scatter(symbols_n.real, symbols_n.imag)

plt.figure(6)
plt.subplot(2,1,1)
plt.plot(np.real(rx_symbols_conv))
plt.plot(np.imag(rx_symbols_conv))
plt.subplot(2,1,2)
plt.scatter(rx_symbols_conv.real, rx_symbols_conv.imag)

plt.figure(7)
plt.subplot(2,1,1)
plt.plot(np.real(rx_down_symbols))
plt.plot(np.imag(rx_down_symbols))
plt.subplot(2,1,2)
plt.scatter(rx_down_symbols.real, rx_down_symbols.imag)

plt.figure(8)
plt.plot(dem_symb)

plt.show()