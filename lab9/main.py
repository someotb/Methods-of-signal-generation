import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import convolution_matrix, inv


def base_signal_output(signal: complex, label: str):
    plt.figure(figsize=(8, 8), label=label)
    plt.subplot(2, 1, 1)
    plt.plot(signal.real, label="I")
    plt.plot(signal.imag, label="Q")
    plt.subplot(2, 1, 2)
    plt.scatter(signal.real, signal.imag, label="I/Q")


def modulator(mod_type: str, bits):
    match mod_type:
        case "BPSK":
            symbols = np.zeros(len(bits), dtype=complex)
            for i in range(len(bits)):
                b0 = 1.0 - 2.0 * bits[i]
                b1 = 1.0 - 2.0 * bits[i]
                symbols[i] = complex(b0, b1) / np.sqrt(2.0)
            return symbols
        case "QPSK":
            symbols = np.zeros(len(bits) // 2, dtype=complex)
            for i in range(len(bits) // 2):
                b0 = 1.0 - 2.0 * bits[2 * i]
                b1 = 1.0 - 2.0 * bits[2 * i + 1]
                symbols[i] = complex(b0, b1) / np.sqrt(2.0)
            return symbols
        case _:
            return "Unsupported modulation type"


def white_noise(symbols, snr_db):
    cos2 = np.abs(symbols) ** 2
    sig_awg_watts = np.mean(cos2)
    sig_awg_db = 10 * np.log10(sig_awg_watts)
    noise_awg_db = sig_awg_db - snr_db
    noise_awg_watts = 10 ** (noise_awg_db / 10)
    mean_noise = 0

    noise_volts_re = np.random.normal(
        mean_noise, np.sqrt(noise_awg_watts), len(symbols)
    )
    noise_volts_im = np.random.normal(
        mean_noise, np.sqrt(noise_awg_watts), len(symbols)
    )
    n_compl = noise_volts_re + 1j * noise_volts_im
    cs_noisy = symbols + n_compl
    return cs_noisy


SNR = 18
ones = np.ones(10)
h = np.array([0.19 + 1j * 0.56, 0.45 - 1j * 1.28, -0.14 - 1j * 0.53])
tr = [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0]
offset = (len(tr) - 16) // 2
tr_central = tr[offset : len(tr) - offset]
bits = np.random.randint(0, 2, 1000)
symbols = modulator("QPSK", bits)
conv_symb = np.convolve(symbols, h[0:1], mode="same")
n_symb = white_noise(conv_symb, SNR)
rec_symb = np.convolve(n_symb, h[0:3], mode="same")
tr_bpsk = modulator("BPSK", tr_central)
akf_tr_bpsk = np.correlate(tr_bpsk, tr_bpsk, mode="same")
r = np.convolve(tr_bpsk, h, mode="same")
M = convolution_matrix(tr_bpsk, len(h), mode="same")
hLS = inv(np.conj(M.T) @ M) @ np.conj(M.T) @ r

print(h)
print(hLS)

base_signal_output(symbols, "Symbols")
base_signal_output(n_symb, "Noisy Symbols")
plt.figure(label="ИХ")
plt.stem([h[i].real for i in range(len(h))])
base_signal_output(rec_symb, "Rec Symbols")
base_signal_output(tr_bpsk, "Tr BPSK")

plt.figure(figsize=(8, 8))
plt.plot(np.abs(akf_tr_bpsk), label="AKF")

plt.show()
