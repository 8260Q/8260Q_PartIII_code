#Toy model

import numpy as np
from Constant_Interaction import CI
import scipy.signal as sig
import matplotlib.pyplot as plt


h_bar = 1.0545718176461565
m_e = 9.1093837015

def e_config(n):
    N = np.zeros(((n+1)**2,2))
    for i in range((n+1)**2):
        N[i, 0] = i//(n+1)
        N[i, 1] = i%(n+1)

    return N

def FDE(n, b_L, b_R):
    N = e_config(n)
    E = np.zeros(np.shape(N)[0])
    for i in range(np.shape(N)[0]):
        E[i] = h_bar**2/m_e*(((N[i, 0]//2)**2 + (N[i, 0]%2)*N[i, 0]/2)*b_L + ((N[i, 1]//2)**2 + (N[i, 1]%2)*N[i, 1]/2)*b_R)

    return E


def generate_stab_mats(der, C, b_L, b_R, res, V_lowL, V_highL, V_lowR, V_highR):
    e_single = FDE(20, b_L, b_R)

    V_L = np.linspace(V_lowL, V_highL, res)
    V_R = np.linspace(V_lowR, V_highR, res)

    E = np.zeros((res, res))
    occ = np.zeros((res, res))
    occ1 = np.zeros((res, res))

    for i in range(res):
        for j in range(res):
            N, eng = CI.CI_E(20, V_L[i], V_R[j], C)
            eng = eng+e_single
            E[i, j] = np.min(eng)
            occ[i, j] = np.argmin(eng)
            occ1[i, j] = np.sum(N[np.argmin(eng), :])

    occ = np.sqrt(np.sum(np.power(np.gradient(occ), 2), axis=0))
    for i in range(res):
        for j in range(res):
            if (occ[i, j] != 0.):
                occ[i, j] = 1.

    gblur = np.array(
        [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256

    #broaden = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

    #occ = sig.convolve2d(occ, broaden, mode="same")
    occ = sig.convolve2d(occ, gblur, mode="same")
    if der==True:
        return occ
    return  occ1

"""
Cgen = CI.Cap(8.4, 20, 4, 4, 4)
C_meas = CI.Cap(30., 30., 10., 4., 4.)
res = 200
#kernel = np.ones((10, 10))

#generated = generate_stab_mats(True, Cgen, 0., 0., 200, 0., 0.4)
measured = generate_stab_mats(True, C_meas, 0.2, 0.3, 200, 0., 0.6, 0., 0.6)

#generated = sig.convolve2d(generated, kernel, mode="same")
#measured = sig.convolve2d(measured, kernel, mode="same")

V = np.linspace(0., 0.6, 200)
kernel = np.ones((15, 15))

meas = sig.convolve2d(measured, kernel, mode="same")

plt.pcolormesh(V, V, meas)
plt.show()
"""
