import numpy as np
from pathlib import Path
from Stability import Fock_Darwin as FD
from Constant_Interaction import CI
import matplotlib.pyplot as plt

path = Path(r"C:\Users\petro\Desktop\Reseach\Data\Statistics\INT_LIN")
save_path = Path(r"C:\Users\petro\Desktop\Reseach\Plots\FD\main Loss functions\FS\predicted_vs_measured_CSD")

lf = np.load(path/"INT_LIN_lf.npy")
y = np.load(path/"INT_LIN_generated.npy")
x = np.load(path/"INT_LIN_measured.npy")

conv = np.zeros(np.shape(lf))

"""
for i in range(np.shape(lf)[0]):
    for j in range(np.shape(lf)[1]):
        conv[i, j] = np.min(lf[i, 0:j+1])

iter_step = np.arange(np.shape(lf)[1])

for i in range(np.shape(lf)[0]):
    plt.plot(iter_step, conv[i, :])
plt.xlabel("Iteration step")
plt.ylabel("Min(Loss function) at nth step")
plt.title("Convergence plot for all GP runs")
plt.show()
"""

x[:, 0] = 30*x[:, 0]
x[:, 1] = 30*x[:, 1]
avg = (x[:, 0] + x[:, 1])/20
x[:, 2] = avg*x[:, 2]
x[:, 3] = avg*x[:, 3]
x[:, 4] = avg*x[:, 4]
x[:, 5] = 0.5*x[:, 5]
x[:, 6] = 0.5*x[:, 6]

y[:, 0] = 30*y[:, 0]
y[:, 1] = 30*y[:, 1]
y[:, 2] = 5*y[:, 2]
y[:, 3] = 5*y[:, 3]
y[:, 4] = 5*y[:, 4]
y[:, 5] = 0.5*y[:, 5]
y[:, 6] = 0.5*y[:, 6]

error = x-y

for i in range(50):
    if 30*np.abs(error[i, 1]) > 10:
        error[i, 1] = 0.
    if 30*np.abs(error[i, 0]) > 10:
        error[i, 0] = 0.

var = np.zeros(7)
avg = np.zeros(7)
ranges = [(-30, 30), (-30, 30), (-8, 8),(-8, 8), (-8, 8), (-0.5, 0.5), (-0.5, 0.5)]

x_lab = ["Cg1", "Cg2", "C_m", "C_L", "C_R", "omegaL", "omegaR"]
for i in range(7):
    var[i] = np.std(error[:, i])
    avg[i] = np.mean(error[:, i])

    plt.hist(error[:, i], bins=75, range=ranges[i])
    plt.ylabel("Frequency")
    plt.xlabel("Error in "+x_lab[i])
    #path_hist = Path(
        #r"C:Users\petro\Desktop\Reseach\Plots\FD\main Loss functions\INT\predicted_vs_measured\error_hist\error_"+x_lab[i]+".png")
    #plt.savefig(path_hist)
    #plt.close()
    plt.show()



print(var)
print(avg)
"""
V = np.linspace(0., 0.4, 200)

for i in range(np.shape(y)[0]):
    C = CI.Cap(y[i, 0],  y[i, 1], y[i, 2], y[i, 3], y[i, 4])
    predicted = FD.generate_stab_mats(True, C, y[i, 5],  y[i, 6], 200, 0., 0.4, 0., 0.4)

    save_path = Path("C:Users\petro\Desktop\Reseach\Plots\FD\main Loss functions\BLUR_post\predicted_vs_measured_CSD\pred_CSD\measured_"+str(i)+".png")

    plt.pcolormesh(V, V, predicted)
    plt.xlabel("Vg1/dVolts")
    plt.ylabel("Vg2/dVolts")

    match i:
        case 0:
            plt.title(str(i + 1) + "st predicted CSD")
        case 1:
            plt.title(str(i + 1) + "nd predicted CSD")
        case 2:
            plt.title(str(i + 1) + "rd predicted CSD")
        case _:
            plt.title(str(i + 1) + "th predicted CSD")

    plt.show()
"""

