import pandas as pd
import numpy as np
import random
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import functions

# Inisiasi
df = pd.read_csv("PEP.csv")

mean_return = functions.mean(df, 1)
std_dev = functions.std(df, 1)

drift = functions.drift(df, 1, 1/252)
volatility = functions.volatility(df, 1, 1/252)


# Metode Pohon Binomial 30 langkah
nominal_interest = 0.025
T = 1/12  # Untuk periode 1 bulan (karena perhitungan dalam 1 tahun)
N = 30  # Banyak langkah
deltat = T/N
S0 = 176.63

drift_bulanan = functions.drift(df, 1, 1/252)
volatility_bulanan = functions.volatility(df, 1, 1/252)
sukubunga_bulanan = functions.r(nominal_interest, 1)
ufactor = functions.u(volatility_bulanan, deltat)
dfactor = 1/ufactor
p = functions.p(sukubunga_bulanan, deltat, ufactor, dfactor)
saham = functions.saham(N, S0, ufactor, dfactor)

print("Menggunakan metode Pohon Binomial 30 langkah, diperoleh :")
functions.opsi("call", N, saham, 115, sukubunga_bulanan, deltat, p, 1)
functions.opsi("call", N, saham, 180, sukubunga_bulanan, deltat, p, 1)
functions.opsi("put", N, saham, 150, sukubunga_bulanan, deltat, p, 1)
functions.opsi("put", N, saham, 185, sukubunga_bulanan, deltat, p, 1)
print("\n")

# Metode Black Scholes
print("Menggunakan metode Black-Scholes, diperoleh: ")
functions.black_scholes(S0, 115, deltat, volatility_bulanan, sukubunga_bulanan, "call", 1)
functions.black_scholes(S0, 180, deltat, volatility_bulanan, sukubunga_bulanan, "call", 1)
functions.black_scholes(S0, 150, deltat, volatility_bulanan, sukubunga_bulanan, "put", 1)
functions.black_scholes(S0, 185, deltat, volatility_bulanan, sukubunga_bulanan, "put", 1)
print("\n")

# Metode Monte-Carlo

