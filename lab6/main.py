from cProfile import label
from tkinter import Label

import numpy as np
from matplotlib import pyplot as plt

S1 = 0.4
S2 = 0
var = 0.04
z_list = [0.3, 0.2, 0.1]


def get_w(r, S):
    return (np.exp(-((r - S) ** 2) / (2 * var))) / (np.sqrt(2 * np.pi * var))


# Task 1
print("Task 1")
for z in z_list:
    w1 = get_w(z, S1)
    w0 = get_w(z, S2)

    Lambda = w1 / w0

    print(f"Отсчет z = {z}:")
    print(f"  w(z|S1) = {w1:.4f}, w(z|S2) = {w0:.4f}")
    print(f"  Отношение правдоподобия Lambda = {Lambda:.4f}")

    if Lambda > 1:
        print("  Решение: ПЕРЕДАНА 1")
    elif Lambda < 1:
        print("  Решение: ПЕРЕДАН 0")
    else:
        print("  Решение: Граница (0 или 1)")
    print("\n")

# Task 2
print("Task 2")
Am = [0.25, 1, 2, 4]
R = 100
N0_2 = 0.25 * 10 ** (-2)

B = R
sigma_sq = 2 * N0_2 * B
print(f"Рассчитанная дисперсия шума (var): {sigma_sq:.4f}\n")

for am in Am:
    S1 = am
    S2 = 0
    s_t = np.random.choice([S1, S2])

    noise = np.random.normal(0, np.sqrt(sigma_sq))
    r_t = s_t + noise

    w1 = get_w(r_t, S1)
    w2 = get_w(r_t, S2)

    Lambda = w1 / w2
    print(f"  Отношение правдоподобия Lambda = {Lambda:.4f}")

    if Lambda > 1:
        print("  Решение: ПЕРЕДАНА 1")
    elif Lambda < 1:
        print("  Решение: ПЕРЕДАН 0")
    else:
        print("  Решение: Граница (0 или 1)")
    print("\n")

# Task 3
print("Task 3")
var = 1.5
S1 = 1
S2 = -S1
bits = np.random.randint(0, 2, 10)
symbols = np.array([2 * b - 1 for b in bits])
noise = np.random.normal(0, np.sqrt(var), size=len(symbols))
noise_symbols = symbols + noise

decision_bits_srav = np.zeros_like(noise_symbols)
decision_bits_otnosh = np.zeros_like(noise_symbols)

for i in range(len(noise_symbols)):
    w1 = get_w(noise_symbols[i], S1)
    w2 = get_w(noise_symbols[i], S2)

    Lambda = w1 / w2
    print(f"  Отношение правдоподобия Lambda = {Lambda:.4f}")
    if Lambda > 1:
        decision_bits_otnosh[i] = 1
    elif Lambda < 1:
        decision_bits_otnosh[i] = 0
    else:
        decision_bits_otnosh[i] = np.random.choice([0, 1])

for i in range(len(noise_symbols)):
    if noise_symbols[i] > ((S1 + S2) / 2):
        decision_bits_srav[i] = 1
    elif noise_symbols[i] < ((S1 + S2) / 2):
        decision_bits_srav[i] = 0
    else:
        continue

plt.figure(1, label="Сравнение отправленных и полученных битов")
plt.plot(bits + 0.2, label="TX (Отправленные)", linewidth=2)
plt.plot(decision_bits_srav + 0.1, label="RX сравнение", linestyle="--")
plt.plot(decision_bits_otnosh, label="RX правдоподобие", linestyle=":")
plt.legend()

# Task 4
print("\nTask 4")
cnt_err = 0
for i in range(len(decision_bits_otnosh)):
    if decision_bits_otnosh[i] != bits[i]:
        cnt_err += 1

BER = cnt_err / len(bits)
print(f"BER: {BER}")

# Task 5
Lambda_list = []
S1 = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5])
S2 = np.array([-i for i in S1])
BER_list = np.array([])

for s in range(len(S1)):
    err_cnt = 0
    dem_bits_t5 = np.zeros_like(bits)
    noise = np.random.normal(0, var, size=len(bits))
    ns_sym = symbols + noise
    for i in range(len(ns_sym)):
        w1 = get_w(ns_sym[i], S1[s])
        w2 = get_w(ns_sym[i], S2[s])

        Lambda = w1 / w2
        Lambda_list.append(Lambda)

        if Lambda > 1:
            dem_bits_t5[i] = 1
        elif Lambda < 1:
            dem_bits_t5[i] = 0
        else:
            dem_bits_t5[i] = np.random.choice([0, 1])

    for b in range(len(dem_bits_t5)):
        if dem_bits_t5[b] != bits[b]:
            err_cnt += 1

    BER = err_cnt / len(bits)
    BER_list = np.append(BER_list, BER)

plt.figure(2, label="BER от Sn")
plt.plot(
    S1,
    BER_list,
)
plt.xlabel("S(n)")
plt.ylabel("BER")

plt.show()
