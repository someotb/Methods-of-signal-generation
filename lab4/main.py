import numpy as np
from matplotlib import pyplot as plt


def get_pdf(xn, left, right, step):
    bins_edges = np.arange(left, right + step, step)
    bins_centers = bins_edges[:-1] + step / 2
    bins_counters = np.zeros_like(bins_centers)

    norv_val = len(xn) * step
    for x in xn:
        for i in range(len(bins_edges) - 1):
            if bins_edges[i] <= x < bins_edges[i + 1]:
                bins_counters[i] += 1
                break

    bins_counters /= norv_val
    return bins_counters, bins_centers


# Task 1
m = 0
var = 3
s = np.sqrt(var)
t = np.linspace(m, var, 3000)

xn = np.random.uniform(m, s, len(t))
step = 0.2

bins_counters, bins_centers = get_pdf(xn, np.min(xn), np.max(xn), step)

plt.figure(1, label="Гистограмма распределения(Xn)")
plt.title("Гистограмма распределения(Xn)")
plt.tight_layout()
plt.bar(bins_centers, bins_counters, step)


Yn = []
for i in range(1, 10001):
    Yn.append(np.sum(np.random.uniform(m, s, len(t))))

bins_counters_yn, bins_centers_yn = get_pdf(Yn, np.min(Yn), np.max(Yn), step)

plt.figure(2, label="Гистограмма распределения(Yn)")
plt.title("Гистограмма распределения(Yn)")
plt.tight_layout()
plt.bar(bins_centers_yn, bins_counters_yn, step)

# Task 2
xn1 = []
sechenie = []
for i in range(10000):
    xn = np.random.normal(m, s, len(t))
    xn1.append(np.convolve(xn, [1, 0.7, 0.3, 0.1, 0.05]))

for i in range(len(xn1)):
    sechenie.append(xn1[i][40])

plt.figure(3, label="Сечения")
plt.title("Сечения")
plt.tight_layout()
plt.plot(sechenie)

bins_counters_sech, bins_centers_sech = get_pdf(
    sechenie, np.min(sechenie), np.max(sechenie), step
)

plt.figure(4, label="Гистограмма распределения сечения")
plt.title("Гистограмма распределения сечения")
plt.tight_layout()
plt.bar(bins_centers_sech, bins_counters_sech, step)

offset = [0, 3, 5, 7]
Bx = []
for i in range(len(xn1)):
    for j in offset:
        Bx.append(xn1[i][40] * xn1[i][40 + j])

Bx_0 = np.sum(Bx[0 :: len(offset)]) / len(xn1)
Bx_3 = np.sum(Bx[1 :: len(offset)]) / len(xn1)
Bx_5 = np.sum(Bx[2 :: len(offset)]) / len(xn1)
Bx_7 = np.sum(Bx[3 :: len(offset)]) / len(xn1)

akf_val = [Bx_0, Bx_3, Bx_5, Bx_7]

plt.figure(5, label="АКФ")
plt.title("АКФ")
plt.tight_layout()
plt.plot(offset, akf_val)

interv_corr = np.sum(akf_val) / akf_val[0]
print(f"Интервал корреляции: {interv_corr}")

# Task 3
xn_1_ed = xn1[0]
B_n = []
for of in offset:
    tmp = 0
    for i in range(len(xn_1_ed) - len(offset)):
        tmp += xn_1_ed[i] * xn_1_ed[i - of]
    B_n.append(tmp / (len(xn_1_ed) - len(offset)))

plt.figure(6, label="АКФ по одной реализации")
plt.title("АКФ")
plt.tight_layout()
plt.plot(offset, B_n)

plt.show()
