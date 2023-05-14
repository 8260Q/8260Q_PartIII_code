from skopt import gp_minimize
from skopt import plots
import numpy as np
from Constant_Interaction import CI
from Stability import Fock_Darwin as FD
from pathlib import Path
import LF
import matplotlib.pyplot as plt
import pandas as pd
from Utilities import interpolate as ip
from PIL import Image
import scipy.signal as sig

path_D1 = Path(r"C:\Users\petro\Desktop\Cambridge\4th year undergrad (Masters)\Project\Learning qubit Hamiltonians from charge stability diagrams\Code\Data_set\Data_set2\M.xlsx")
path = Path(r"C:\Users\petro\Desktop\Reseach\Data\Experimental")

g = pd.read_excel(io=path_D1, sheet_name=0)
Vg1 = pd.read_excel(io=path_D1, sheet_name=1)
Vg2 = pd.read_excel(io=path_D1, sheet_name=2)

g.to_numpy()
Vg1.to_numpy()
Vg2.to_numpy()

VL = 10*Vg1[0]
VR = Vg2[0]

V_lowL = VL[0]
V_highL = VL[len(VL)-1]

V_lowR = VR[0]
V_highR = VR[len(VR)-1]
res = np.shape(g)


"""
for i in range(res[0]):
    for j in range(res[1]):
        if np.abs(g[j][i]) < 0.05*np.max(np.abs(g)):
            g[i][j] = 0.
        else:
            g[i][j] = 1.

gblur = np.array(
        [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256
y_broaden = np.array([[1], [1], [1], [1]])
CSD = sig.convolve2d(CSD, y_broaden, mode="same")
CSD = sig.convolve2d(CSD, gblur, mode="same")
"""

V_L = np.linspace(V_lowL, V_highL, res[0])
V_R = np.linspace(V_lowR, V_highR, res[1])

plt.pcolormesh(V_R, V_L, g, shading="nearest")
plt.xlabel("Vg1/dV")
plt.ylabel("Vg2/dV")
plt.title("Experimentally measured CSD")
plt.show()

"""
ranges = [(0.0001, 1.), (0.0001, 1.), (0.0001, 1.), (0.0001, 1.), (0.0001, 1.), (0.0001, 1.), (0.0001, 1.)]

res = gp_minimize(
                lambda x: LF.INT_LIN(x, measured=CSD, res=res, V_lowL=V_lowL,
                                    V_highL=V_highL, V_lowR=V_lowR, V_highR=V_highR), ranges, n_calls=40,
                                    n_random_starts=15, acq_func="LCB")

f_vals = res.func_vals
x_opt = res.x
x_vals = res.x_iters

np.save(path / 'x_vals', x_vals)
np.save(path / 'x_opt', x_opt)
np.save(path / 'f_vals', f_vals)

FD_plot = plots.plot_objective(res)
plt.show()
"""
