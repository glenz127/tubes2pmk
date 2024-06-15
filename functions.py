import numpy as np
import math
from scipy.stats import norm
import random
import matplotlib.pyplot as plt


def mean(df, periode):
    """
    Fungsi ini mencari nilai mean untuk mencari nilai parameter drift nantinya
    :param df:
    :param periode:
    :return:
    """
    df2 = df.iloc[::periode, :].reset_index(drop=True)
    df2["Return"] = np.log(df2["Adj Close"]/df2["Adj Close"].shift(1))
    mean_return = df2["Return"].sum()/(len(df2)-1)
    return mean_return


def std(df, periode):
    """
    Fungsi ini digunakan untuk mencari nilai standar deviasi dari saham untuk mencari
    parameter volatilitas
    :param df:
    :param periode:
    :return:
    """
    df2 = df.iloc[::periode, :].reset_index(drop=True)
    mu = mean(df, periode)
    df2["Return"] = np.log(df2['Adj Close']/df2['Adj Close'].shift(1))
    df2["SS"] = (df2["Return"] - mu)**2
    st_dev = np.sqrt((1/(len(df2["Return"])-2))*(df2["SS"].sum()))
    return st_dev


def drift(df, periode, T):
    """
    Fungsi ini mencari nilai parameter drift
    :param df:
    :param periode:
    :param T:
    :return:
    """
    drift_value = (1/T)*(mean(df, periode) + 0.5*std(df, periode)**2)
    return drift_value


def volatility(df, periode, T):
    """
    Fungsi ini mencari nilai parameter volatility
    :param df:
    :param periode:
    :param T:
    :return:
    """
    volatility_value = std(df, periode)/np.sqrt(T)
    return volatility_value


def r(bunga, periode):
    i = bunga/(12/periode)
    return i


def u(sigma, deltat):
    """ Fungsi yang menghitung increase factor(u) """
    return np.exp(sigma*np.sqrt(deltat))


def p(interest, deltat, u_factor, d_factor):
    """ Fungsi yang menghitung peluang naik"""
    return (np.exp(interest*deltat)-d_factor)/(u_factor-d_factor)


def saham(N, S0, u, d):
    S = np.zeros((N+1, N+1))
    S[0][0] = S0
    for i in range(0, N+1):
        for j in range(0, i+1):
            S[i][j] = S0 * u**j * d**(i-j)
    return S


def opsi(tipe, N, S, K, interest, deltat, p, t):
    """
    Fungsi ini menghitung harga opsi
    :param tipe:
    :param N:
    :param S:
    :param K:
    :param interest:
    :param deltat:
    :param p:
    :param t:
    :return:
    """
    if tipe.lower() == "call":
        """
        Bagian Ini menghitung harga opsi Call
        """
        value = np.zeros((N+1, N+1))
        for i in range(N+1):
            value[N][i] = np.round(np.maximum(S[N][i]-K, 0), 4)
        for i in reversed(range(N)):
            for j in range(0, i+1):
                value[i][j] = np.round(np.exp(-interest*deltat)*(p*value[i+1][j+1] + (1-p)*value[i+1][j]), 4)
        harga_opsi = value[0][0]
        print(f"Harga Opsi {tipe} dengan Strike Price {K} dan waktu jatuh tempo {t} bulan adalah {harga_opsi}")
    elif tipe.lower() == "put":
        """
        Bagian ini menghitung harga opsi Put
        """
        value = np.zeros((N+1, N+1))
        for i in range(N+1):
            value[N][i] = np.round(np.maximum(K-S[N][i], 0), 4)
        for i in reversed(range(N)):
            for j in range(0, i+1):
                value[i][j] = np.round(np.exp(-interest*deltat)*(p*value[i+1][j+1] + (1-p)*value[i+1][j]), 4)
        harga_opsi = value[0][0]
        print(f"Harga Opsi {tipe} dengan Strike Price {K} dan waktu jatuh tempo {t} bulan adalah {harga_opsi}")
    return


def black_scholes(opsi, S, K, deltat, sigma, interest, t):
    """
    Fungsi ini menghitung harga opsi put/call menggunakan metode Black-Scholes dengan informasi
     data harian selama 1 tahun, untuk opsi yang jatuh tempo dalam t bulan
    :param S:
    :param K:
    :param deltat:
    :param sigma:
    :param interest:
    :param opsi:
    :param t:
    :return:
    """
    d1 = (math.log(S/K) + (interest + sigma**2/2)*deltat) / (sigma * math.sqrt(deltat))
    d2 = d1 - sigma*math.sqrt(deltat)
    if opsi.lower() == "call":
        C = S*norm.cdf(d1) - K*math.exp(-interest*deltat)*norm.cdf(d2)
        print(f"Harga Opsi {opsi} dengan Strike Price {K} dan waktu jatuh tempo {t} "
              f"bulan adalah {np.round(C, 4)}")
    elif opsi.lower() == "put":
        P = K*math.exp(-interest*deltat)*norm.cdf(-d2) - S*norm.cdf(-d1)
        print(f"Harga Opsi {opsi} dengan Strike Price {K} dan waktu jatuh tempo {t} "
              f"bulan adalah {np.round(P, 4)}")


def monte_carlo(tipe, S0, K, t, N, deltat, mu, sigma, interest):
    S_t_all = np.zeros((N+1, N+1))
    S_t = np.zeros(N+1)
    S_t[0] = S0
    for i in range(len(S_t_all)):
        S_t_all[i, 0]= S0
    payoff = np.zeros(N+1)

    for i in range(1, N+1):
        for j in range(1, N+1):
            z = np.random.standard_normal()
            S_t[j] = S_t[j-1] * np.exp((mu-0.5*sigma**2)*deltat + sigma*np.sqrt(deltat)*z)
            S_t_all[i, j] = S_t[j]
        if tipe == "call":
            payoff[i] = np.maximum(S_t_all[i, N-1]-K, 0)
        elif tipe == "put":
            payoff[i] = np.maximum(K - S_t_all[i, N-1], 0)
    harga_opsi = np.round(np.exp(-interest*deltat) * np.mean(payoff), 4)

    print(f"Harga opsi {tipe} dengan periode {t} bulan dan Strike Price {K} adalah {harga_opsi}.")
    return








