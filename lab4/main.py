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
plt.plot(bins_centers, bins_counters)


Yn = []
for i in range(1, 10001):
    Yn.append(np.sum(np.random.uniform(m, s, len(t))))

bins_counters_yn, bins_centers_yn = get_pdf(Yn, np.min(Yn), np.max(Yn), step)

plt.figure(2, label="Гистограмма распределения(Yn)")
plt.title("Гистограмма распределения(Yn)")
plt.tight_layout()
plt.stem(bins_centers_yn, bins_counters_yn)

plt.show()
