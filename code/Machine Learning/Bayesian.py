from skopt import gp_minimize
from skopt import plots
from skopt.space import Real
import numpy as np
from Constant_Interaction import CI
from Stability import Fock_Darwin as FD
from pathlib import Path
import LF
import matplotlib.pyplot as plt
import pandas as pd


path = Path(r"C:\Users\petro\Desktop\Reseach\Data\Statistics\INT_LIN")
path_post = Path(r"C:\Users\petro\Desktop\Reseach\Data\Statistics\BLUR_post")
path_D1 = Path(r"C:\Users\petro\Desktop\Cambridge\4th year undergrad (Masters)\Project\Learning qubit Hamiltonians from charge stability diagrams\Code\Data_set\Data_set1")


C_measured = CI.Cap(21.2, 15.4, 3.1, 4.2, 1.1)
V_low = 0.
V_high = 0.6
res = 200
b_L = 0.22
b_R = 0.31

measured = FD.generate_stab_mats(True, C_measured, b_L, b_R, res, V_low, V_high, V_low, V_high)


ranges_FD = [[(0.033, 1.), (0.033, 1.), (0.0001, 1.), (0.0001, 1.), (0.0001, 1.), (0.0001, 1.), (0.0001, 1.)],
            [(0.0001, 1.), (0.0001, 1.), (0.0001, 1.), (0.0001, 1.), (0.0001, 1.)],
            [(0.0001, 1.), (0.0001, 1.), (0.0001, 1.)]]
"""
init_cond = np.zeros(7)


for i in range(4):
    match i:
        case 0:
            res = gp_minimize(
                lambda x: LF.INT_LIN_FD(x, opt_cycle=i, init=init_cond, measured=measured_der, res=res, V_low=V_low,
                                    V_high=V_high), ranges_FD[i], n_calls=100, n_random_starts=30, acq_func="LCB")
            f_vals = res.func_vals
            x_vals = res.x
            init_cond[0] = x_vals[0]
            init_cond[1] = x_vals[1]
        case 1:
            res = gp_minimize(
                lambda x: LF.INT_LIN_FD(x, opt_cycle=i, init=init_cond, measured=measured_der, res=res, V_low=V_low,
                                    V_high=V_high), ranges_FD[i], n_calls=100, n_random_starts=30, acq_func="LCB")
            f_vals = res.func_vals
            x_vals = res.x
            init_cond[5] = x_vals[3]
            init_cond[6] = x_vals[4]
        case 2:
            res = gp_minimize(
                lambda x: LF.INT_LIN_FD(x, opt_cycle=i, init=init_cond, measured=measured_der, res=res, V_low=V_low,
                                    V_high=V_high), ranges_FD[i], n_calls=100, n_random_starts=30, acq_func="LCB")
            f_vals = res.func_vals
            x_vals = res.x
            init_cond[2] = x_vals[0]
            init_cond[3] = x_vals[1]
            init_cond[4] = x_vals[2]
        case 3:
            ranges = [(0.9*init_cond[0], 1.1*init_cond[0]), (0.9*init_cond[1], 1.1*init_cond[1])
                , (0.9*init_cond[2], 1.1*init_cond[2]), (0.9*init_cond[3], 1.1*init_cond[3])
                , (0.9*init_cond[4], 1.1*init_cond[4]), (0.9*init_cond[5], 1.1*init_cond[5])
                , (0.9*init_cond[6], 1.1*init_cond[6])]

            res = gp_minimize(
                lambda x: LF.INT_LIN_FD(x, opt_cycle=i, init=init_cond, measured=measured_der, res=res, V_low=V_low, V_high=V_high),
                ranges, n_calls=100, n_random_starts=30, acq_func="LCB")

            np.save(path_FD / 'INT_grad_x', res.x_iters)
            np.save(path_FD / 'INT_grad_lf', res.func_vals)

            FD_plot = plots.plot_objective(res)
            plt.show()

"""

ranges = [Real(1., 30., name="Cg1"), Real(1., 30., name="Cg2"), Real(0.001, 8., name="Cm"), Real(0.001, 8., name="CL"),
          Real(0.001, 8., name="CR"), Real(0.0001, 0.5, name="βL"), Real(0.0001, 0.5, name="βR")]


#FD_FS = gp_minimize(lambda x: LF.FS_FD(x, measured=measured_der, res=res, V_low_L=V_low, V_high_L=V_high, V_low_R=V_low, V_high_R=V_high), ranges, n_calls=120, n_random_starts=30, acq_func="LCB")
#FD_BLUR = gp_minimize(lambda x: LF.BLUR_FD(x, measured=measured_der, res=res, V_low_L=V_low, V_high_L=V_high, V_low_R=V_low, V_high_R=V_high), ranges, n_calls=120, n_random_starts=30, acq_func="LCB")
FD_INT = gp_minimize(lambda x: LF.INT_LIN(x, measured=measured, res=res, V_lowL=V_low, V_highL=V_high, V_lowR=V_low, V_highR=V_high), ranges, n_calls=120, n_random_starts=30, acq_func="LCB")

#FS_plot = plots.plot_objective(FD_FS)
#BLUR_plot = plots.plot_objective(FD_BLUR)
INT_plot = plots.plot_objective(FD_INT)

plt.show()

"""
res = 200
V_low = 0.
V_high = 0.4

y = np.zeros((50, 7))
lf = np.zeros((50, 30))
INT_pred = np.load(path/"INT_LIN_generated.npy")
INT_meas = np.load(path/"INT_LIN_measured.npy")

for i in range(50):
    init = INT_pred[i, :]
    Cg1_r = [0.95*30*init[0], 1.05*30*init[0]]
    Cg2_r = [0.95 * 30*init[1], 1.05 * 30*init[1]]
    ranges = [Real(Cg1_r[0], Cg1_r[1], name="Cg1"), Real(Cg2_r[0], Cg2_r[1], name="Cg2"), Real(0.001, 8., name="Cm"),
              Real(0.001, 8., name="CL"),
              Real(0.001, 8., name="CR"), Real(0.0001, 0.5, name="βL"), Real(0.0001, 0.5, name="βR")]

    Cg1 = 30*INT_meas[:, 0]
    Cg2 = 30 * INT_meas[:, 1]
    avg = 3*(Cg1[i] + Cg2[i])/2
    C_m = INT_meas[:, 2]
    CCL = INT_meas[:, 3]
    CCR = INT_meas[:, 4]
    b_L = 0.5 * INT_meas[:, 5]
    b_R = 0.5 * INT_meas[:, 6]


    C_measured = CI.Cap(Cg1[i], Cg2[i], avg*C_m[i], avg*CCL[i], avg*CCR[i])

    measured = FD.generate_stab_mats(True, C_measured, b_L[i], b_R[i], res, V_low, V_high, V_low, V_high)

    BLUR_post = gp_minimize(lambda x: LF.INT_LIN(x, measured=measured, res=res, V_lowL=V_low, V_highL=V_high, V_lowR=V_low, V_highR=V_high), ranges, n_calls=30, n_random_starts=20, acq_func="LCB")

    lf[i, :] = BLUR_post.func_vals
    y[i, :] = BLUR_post.x

np.save(path/"post_predicted", y)
np.save(path/"post_lf", lf)
"""
