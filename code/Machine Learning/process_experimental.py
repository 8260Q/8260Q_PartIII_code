import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from Stability import Fock_Darwin as FD
from Constant_Interaction import CI
from Utilities import interpolate as ip
import pandas as pd
import scipy.signal as sig

path = Path(r"C:\Users\petro\Desktop\Reseach\Data\Experimental")
path_meas = Path(r"C:\Users\petro\Desktop\Cambridge\4th year undergrad (Masters)\Project\Learning qubit Hamiltonians from charge stability diagrams\Code\Data_set\Data_set2\M.xlsx")

g = pd.read_excel(io=path_meas, sheet_name=0)
Vg1 = pd.read_excel(io=path_meas, sheet_name=1)
Vg2 = pd.read_excel(io=path_meas, sheet_name=2)

g.to_numpy()
Vg1.to_numpy()
Vg2.to_numpy()

VL = 10*Vg1[0]
VR = Vg2[0]

V_lowL = VL[0]
V_highL = VL[len(VL)-1]

V_lowR = VR[0]
V_highR = VR[len(VR)-1]
"""
res = 200
CSD = ip.rescale(g, res)

for i in range(res):
    for j in range(res):
        if np.abs(CSD[i][j]) < 0.05*np.max(np.abs(CSD)):
            CSD[i][j] = 0.
        else:
            CSD[i][j] = 1.

gblur = np.array(
        [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256

y_broaden = np.array([[1], [1], [1], [1]])
CSD = sig.convolve2d(CSD, y_broaden, mode="same")

for i in range(res):
    for j in range(res):
        if np.abs(CSD[i][j]) < 0.05*np.max(np.abs(CSD)):
            CSD[i][j] = 0.
        else:
            CSD[i][j] = 1.

CSD = sig.convolve2d(CSD, gblur, mode="same")
"""
V_L = np.linspace(V_lowL, V_highL, 200)
V_R = np.linspace(V_lowR, V_highR, 200)

lf = np.load(path/"f_vals.npy")
y = np.load(path/"x_opt.npy")
print(y)

C = CI.Cap(30*y[0], 30*y[1], 8*y[2], 8*y[3], 8*y[4])

conv = np.zeros(len(lf))

for i in range(len(lf)):
    conv[i] = np.min(lf[0:i+1])

iter_step = np.arange(len(lf))

pred = FD.generate_stab_mats(True, C, 0.5*y[5], 0.5*y[6], 200, 0., 1., 0., 1.)
"""
plt.plot(iter_step, conv)
plt.xlabel("Iteration step")
plt.ylabel("Min(Loss function) at nth step")
plt.title("Convergence plot for predicting experimental CSD")
plt.show()

plt.pcolormesh(V_L, V_R, CSD)
plt.title("Measured CSD")
plt.xlabel("Vg1")
plt.ylabel("Vg2")
plt.show()
"""
V = np.linspace(0., 1., 200)
plt.pcolormesh(V, V, pred)
plt.title("Predicted CSD")
plt.xlabel("Vg1")
plt.ylabel("Vg2")
plt.show()

