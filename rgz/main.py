import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import convolution_matrix, inv

# Параметры варианта: Любимов Кирилл, ИА-331
MODULATION = "QPSK"
TS_LENGTH = 16
NUM_BITS = 1000
CHANNEL_TAPS = 3
SNR_RANGE = np.arange(0, 21, 2)


def modulator(mod_type: str, bits):
    """Модулятор для BPSK/QPSK"""
    match mod_type:
        case "BPSK":
            symbols = np.zeros(len(bits), dtype=complex)
            for i in range(len(bits)):
                b0 = 1.0 - 2.0 * bits[i]
                symbols[i] = complex(b0, 0) / np.sqrt(1.0)
            return symbols
        case "QPSK":
            symbols = np.zeros(len(bits) // 2, dtype=complex)
            for i in range(len(bits) // 2):
                b0 = 1.0 - 2.0 * bits[2 * i]
                b1 = 1.0 - 2.0 * bits[2 * i + 1]
                symbols[i] = complex(b0, b1) / np.sqrt(2.0)
            return symbols
        case _:
            raise ValueError(f"Unsupported modulation type: {mod_type}")


def demodulator(mod_type: str, symbols):
    """Демодулятор для BPSK/QPSK"""
    match mod_type:
        case "BPSK":
            bits = np.zeros(len(symbols), dtype=int)
            for i in range(len(symbols)):
                bits[i] = 0 if symbols[i].real > 0 else 1
            return bits
        case "QPSK":
            bits = np.zeros(len(symbols) * 2, dtype=int)
            for i in range(len(symbols)):
                bits[2 * i] = 0 if symbols[i].real > 0 else 1
                bits[2 * i + 1] = 0 if symbols[i].imag > 0 else 1
            return bits
        case _:
            raise ValueError(f"Unsupported modulation type: {mod_type}")


def generate_training_sequence(length):
    """Генерация тренировочной последовательности с хорошей автокорреляцией"""
    ts_bits = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0])
    return ts_bits[:length]


def white_noise(symbols, snr_db):
    """Добавление белого гауссовского шума"""
    signal_power = np.mean(np.abs(symbols) ** 2)
    signal_power_db = 10 * np.log10(signal_power)
    noise_power_db = signal_power_db - snr_db
    noise_power = 10 ** (noise_power_db / 10)

    noise_re = np.random.normal(0, np.sqrt(noise_power / 2), len(symbols))
    noise_im = np.random.normal(0, np.sqrt(noise_power / 2), len(symbols))
    noise = noise_re + 1j * noise_im

    return symbols + noise


def channel_ls_estimation(ts_symbols, received_ts, channel_length):
    """LS-оценка импульсной характеристики канала"""
    C = convolution_matrix(ts_symbols, channel_length, mode='same')
    h_ls = inv(np.conj(C.T) @ C) @ np.conj(C.T) @ received_ts
    return h_ls


def zero_forcing_equalizer(h_est, signal_length):
    """ZF эквалайзер в частотной области"""
    # Переход в частотную область
    H = np.fft.fft(h_est, n=signal_length)
    # ZF: W = 1/H
    W = 1.0 / (H + 1e-10)
    return W


def mmse_equalizer(h_est, snr_db, signal_length):
    """MMSE эквалайзер в частотной области"""
    # Переход в частотную область
    H = np.fft.fft(h_est, n=signal_length)
    # Дисперсия шума
    noise_var = 10 ** (-snr_db / 10)
    # MMSE: W = H*/(|H|^2 + σ^2)
    W = np.conj(H) / (np.abs(H) ** 2 + noise_var)
    return W


def calculate_ber(bits_tx, bits_rx):
    """Расчет BER"""
    errors = np.sum(bits_tx != bits_rx)
    return errors / len(bits_tx)


def simulate_system(snr_db, h_true, use_mmse=False):
    """Симуляция всей системы"""
    # Генерация данных
    data_bits = np.random.randint(0, 2, NUM_BITS)
    data_symbols = modulator(MODULATION, data_bits)

    # Генерация тренировочной последовательности
    ts_bits = generate_training_sequence(TS_LENGTH)
    ts_symbols = modulator("BPSK", ts_bits)

    # Передача TS через канал
    ts_received = np.convolve(ts_symbols, h_true, mode='same')
    ts_received_noisy = white_noise(ts_received, snr_db)

    # Оценка канала
    h_est = channel_ls_estimation(ts_symbols, ts_received_noisy, len(h_true))

    # Передача данных через канал
    data_received = np.convolve(data_symbols, h_true, mode='same')
    data_received_noisy = white_noise(data_received, snr_db)

    # Эквализация в частотной области
    R = np.fft.fft(data_received_noisy)

    if use_mmse:
        W = mmse_equalizer(h_est, snr_db, len(data_received_noisy))
    else:
        W = zero_forcing_equalizer(h_est, len(data_received_noisy))

    # Применение эквалайзера
    S_eq = R * W
    data_equalized = np.fft.ifft(S_eq)

    # Демодуляция
    bits_rx = demodulator(MODULATION, data_equalized)

    # Обрезка до исходной длины
    bits_rx = bits_rx[:len(data_bits)]

    # Расчет BER
    ber = calculate_ber(data_bits, bits_rx)

    return ber, h_est, data_symbols, data_received_noisy, data_equalized


def plot_results(h_true, h_est, symbols, received, equalized, ber_zf, ber_mmse):
    """Построение всех графиков"""
    fig = plt.figure(figsize=(16, 12))

    # 1. Импульсная характеристика канала
    ax1 = plt.subplot(3, 3, 1)
    ax1.stem(np.abs(h_true), label='Истинная h', linefmt='b-', markerfmt='bo')
    ax1.stem(np.abs(h_est), label='Оценка ĥ', linefmt='r--', markerfmt='rx')
    ax1.set_title('Импульсная характеристика канала')
    ax1.set_xlabel('Отсчет')
    ax1.set_ylabel('Амплитуда')
    ax1.legend()
    ax1.grid(True)

    # 2. Созвездие переданного сигнала
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(symbols.real, symbols.imag, alpha=0.5, s=10)
    ax2.set_title('Созвездие переданного сигнала')
    ax2.set_xlabel('I')
    ax2.set_ylabel('Q')
    ax2.grid(True)
    ax2.axis('equal')

    # 3. Созвездие принятого сигнала
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(received.real, received.imag, alpha=0.5, s=10)
    ax3.set_title('Созвездие принятого сигнала (с шумом)')
    ax3.set_xlabel('I')
    ax3.set_ylabel('Q')
    ax3.grid(True)
    ax3.axis('equal')

    # 4. Созвездие после эквализации
    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(equalized.real, equalized.imag, alpha=0.5, s=10)
    ax4.set_title('Созвездие после эквализации')
    ax4.set_xlabel('I')
    ax4.set_ylabel('Q')
    ax4.grid(True)
    ax4.axis('equal')

    # 5. BER от SNR для ZF
    ax5 = plt.subplot(3, 3, 5)
    ax5.semilogy(SNR_RANGE, ber_zf, 'b-o', label='ZF')
    ax5.set_title('BER от SNR (ZF эквалайзер)')
    ax5.set_xlabel('SNR, дБ')
    ax5.set_ylabel('BER')
    ax5.grid(True, which='both')
    ax5.legend()

    # 6. BER от SNR для MMSE
    ax6 = plt.subplot(3, 3, 6)
    ax6.semilogy(SNR_RANGE, ber_mmse, 'r-s', label='MMSE')
    ax6.set_title('BER от SNR (MMSE эквалайзер)')
    ax6.set_xlabel('SNR, дБ')
    ax6.set_ylabel('BER')
    ax6.grid(True, which='both')
    ax6.legend()

    # 7. Сравнение ZF и MMSE
    ax7 = plt.subplot(3, 3, 7)
    ax7.semilogy(SNR_RANGE, ber_zf, 'b-o', label='ZF')
    ax7.semilogy(SNR_RANGE, ber_mmse, 'r-s', label='MMSE')
    ax7.set_title('Сравнение ZF и MMSE')
    ax7.set_xlabel('SNR, дБ')
    ax7.set_ylabel('BER')
    ax7.grid(True, which='both')
    ax7.legend()

    # 8. Фазовая характеристика канала
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(np.angle(h_true), 'b-o', label='Истинная h')
    ax8.plot(np.angle(h_est), 'r--x', label='Оценка ĥ')
    ax8.set_title('Фазовая характеристика канала')
    ax8.set_xlabel('Отсчет')
    ax8.set_ylabel('Фаза, рад')
    ax8.legend()
    ax8.grid(True)

    # 9. Ошибка оценки канала
    ax9 = plt.subplot(3, 3, 9)
    error = np.abs(h_true - h_est)
    ax9.stem(error)
    ax9.set_title('Ошибка оценки канала |h - ĥ|')
    ax9.set_xlabel('Отсчет')
    ax9.set_ylabel('Ошибка')
    ax9.grid(True)

    plt.tight_layout()
    plt.savefig('/home/someotb/Code/Methods-of-signal-generation/rgz/results.png', dpi=300)
    print("Графики сохранены в results.png")


def main():
    """Основная функция"""
    print("=" * 60)
    print("РГЗ: Оценка канала и эквализация")
    print("Студент: Любимов Кирилл, группа ИА-331")
    print("=" * 60)
    print(f"Модуляция: {MODULATION}")
    print(f"Длина TS: {TS_LENGTH}")
    print(f"Количество бит: {NUM_BITS}")
    print(f"Количество лучей канала: {CHANNEL_TAPS}")
    print("=" * 60)

    # Истинная импульсная характеристика канала
    h_true = np.array([0.19 + 1j * 0.56, 0.45 - 1j * 1.28, -0.14 - 1j * 0.53])

    print("\nИстинная импульсная характеристика канала:")
    print(h_true)

    # Симуляция для разных SNR
    ber_zf = []
    ber_mmse = []

    print("\nСимуляция для разных SNR:")
    for snr in SNR_RANGE:
        ber_z, _, _, _, _ = simulate_system(snr, h_true, use_mmse=False)
        ber_m, _, _, _, _ = simulate_system(snr, h_true, use_mmse=True)
        ber_zf.append(ber_z)
        ber_mmse.append(ber_m)
        print(f"SNR = {snr:2d} дБ: BER(ZF) = {ber_z:.6f}, BER(MMSE) = {ber_m:.6f}")

    # Симуляция для построения графиков (SNR = 10 дБ)
    snr_plot = 10
    _, h_est, symbols, received, equalized = simulate_system(snr_plot, h_true, use_mmse=False)

    print(f"\nОценка канала при SNR = {snr_plot} дБ:")
    print(h_est)
    print(f"\nОшибка оценки: {np.linalg.norm(h_true - h_est):.6f}")

    # Построение графиков
    plot_results(h_true, h_est, symbols, received, equalized, ber_zf, ber_mmse)

    plt.show()

    print("\nГотово!")


if __name__ == "__main__":
    main()
