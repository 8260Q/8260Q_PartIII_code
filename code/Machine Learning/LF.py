from Constant_Interaction import CI
from Stability import Fock_Darwin as FD
from Stability import CI_stab
import numpy as np
import scipy.signal as sig


def MSE_FS_CI(x):
    C_meas = CI.Cap(40, 40, 10, 10, 10)
    res = 200
    V_low = 0.
    V_high = 1.
    measured = CI_stab.generate_stab_mats(C_meas, res, 0., 1.)

    C_gen = CI.Cap(x[0] * 80, 40, 10, 10, 10)
    generated = CI_stab.generate_stab_mats(C_gen, res, V_low, V_high)

    ft_m = np.fft.fft2(measured)
    ft_g = np.fft.fft2(generated)

    ft_m = np.log(np.abs(ft_m/np.max(ft_m)) + 0.0001)
    ft_g = np.log(np.abs(ft_g/np.max(ft_g)) + 0.0001)

    diff = ft_m - ft_g

    lf = np.sum(np.power(diff, 2.))/res**2

    return lf


def PD_CI(x):
    C_meas = CI.Cap(40, 40, 10, 10, 10)
    res = 200
    V_low = 0.
    V_high = 1.
    measured = CI_stab.generate_stab_mats(C_meas, res, 0., 1.)

    C_gen = CI.Cap(x[0] * 80, 40, 10, 10, 10)
    generated = CI_stab.generate_stab_mats(C_gen, res, V_low, V_high)

    lf = 0.

    for i in range(res):
        for j in range(res):
            if (measured[i, j] != 0):
                N = 0
                while ((not (generated[i-N:i+N+1, j-N:j+N+1].any())) and N<res):
                    N = N + 1

                lf += N**2

    return lf/res**2


def MSE_CI(x):
    C_meas = CI.Cap(40, 40, 10, 10, 10)
    res = 200
    V_low = 0.
    V_high = 1.
    measured = CI_stab.generate_stab_mats(C_meas, res, 0., 1.)

    C_gen = CI.Cap(x[0] * 80, 40, 10, 10, 10)
    generated = CI_stab.generate_stab_mats(C_gen, res, V_low, V_high)

    diff = generated - measured

    lf = np.sum(np.power(diff, 2.))/res**2

    return lf


def MSE_BLUR_CI(x):
    C_meas = CI.Cap(40, 40, 10, 10, 10)
    res = 200
    V_low = 0.
    V_high = 1.
    measured = CI_stab.generate_stab_mats(C_meas, res, 0., 1.)

    C_gen = CI.Cap(x[0] * 80, 40, 10, 10, 10)

    k = int(res / 20)

    kernel = np.ones((k, k))

    generated = CI_stab.generate_stab_mats(C_gen, res, V_low, V_high)
    generated = sig.convolve2d(generated, kernel, mode="same")
    measured = sig.convolve2d(measured, kernel, mode="same")

    diff = generated - measured

    lf = np.sum(np.power(diff, 2.)) / res ** 2

    return lf


def MSE_INT_CI(x):
    C_meas = CI.Cap(40, 40, 10, 10, 10)
    res = 200
    V_low = 0.
    V_high = 1.
    measured = CI_stab.generate_stab_mats(C_meas, res, 0., 1.)

    C_gen = CI.Cap(x[0] * 80, 40, 10, 10, 10)
    generated = CI_stab.generate_stab_mats(C_gen, res, V_low, V_high)

    int_measured = np.zeros((res, res))
    int_generated = np.zeros((res, res))

    N_m = 0
    N_g = 0
    j = 0
    for i in range(0, res):
        if (measured[i, j] != 0 and measured[i - 1, j] == 0):
            N_m = N_m + 1

        if (generated[i, j] != 0 and generated[i - 1, j] == 0):
            N_g = N_g + 1

        for j in range(0, res):

            temp_m = N_m
            temp_g = N_g

            if (measured[i, j] != 0 and measured[i, j - 1] == 0):
                temp_m = temp_m + 1

            if (generated[i, j] != 0 and generated[i, j - 1] == 0):
                temp_g = temp_g + 1

            int_measured[i, j] = temp_m
            int_generated[i, j] = temp_g

    diff = int_generated - int_measured

    lf = np.sum(np.power(diff, 2.)) / res ** 2

    return lf




def PD_FD(x):
    lf = 0.
    C_meas = CI.Cap(40, 40, 10, 10, 10)
    res = 200
    V_low = 0.
    V_high = 1.
    b_L = 0.2
    b_R = 0.1
    measured = FD.generate_stab_mats(C_meas, b_L, b_R, res, V_low, V_high)

    C_gen = CI.Cap(40, 80*x[0], 10, 10, 10)

    generated = FD.generate_stab_mats(C_gen, x[1], b_R, res, V_low, V_high)

    for i in range(res):
        for j in range(res):
            if (measured[i, j] != 0):
                N = 0
                while ((not (generated[i - N:i + N + 1, j - N:j + N + 1].any())) and N < res):
                    N = N + 1

                lf += N ** 2

    return lf / res ** 2


def MSE_FD(x):
    C_meas = CI.Cap(40, 40, 10, 10, 10)
    res = 200
    V_low = 0.
    V_high = 1.
    b_L = 0.2
    b_R = 0.1
    measured = FD.generate_stab_mats(C_meas, b_L, b_R, res, V_low, V_high)

    C_gen = CI.Cap(40, 80*x[0], 10, 10, 10)

    generated = FD.generate_stab_mats(C_gen, x[1], b_R, res, V_low, V_high)

    diff = generated - measured

    lf = np.sum(np.power(diff, 2.))/res**2

    return lf


def BLUR_FD(x, measured, res, V_low_L, V_high_L, V_low_R, V_high_R):

    C_gen = CI.Cap(x[0], x[1], x[2], x[3], x[4])
    generated = FD.generate_stab_mats(True, C_gen, x[5], x[6], res, V_low_L, V_high_L, V_low_R, V_high_R)

    k = int(res/20)
    kernel = np.ones((k, k))

    generated = sig.convolve2d(generated, kernel, mode="same")
    measured = sig.convolve2d(measured, kernel, mode="same")

    diff = generated - measured

    lf = np.sum(np.power(diff, 2.)) / res ** 2

    return lf


def INT_FD(x, opt_cycle, init, measured, res, V_low, V_high):
    match opt_cycle:
        case 0:
            C_gen = CI.Cap(30 * x[0], 30 * x[1], 8 * x[2], 8 * x[3], 8 * x[4])
            generated = FD.generate_stab_mats(False, C_gen, 0.5 * x[5], 0.5 * x[6], res, V_low, V_high)
        case 1:
            C_gen = CI.Cap(30 * init[0], 30 * init[1], 8 * x[0], 8 * x[1], 8 * x[2])
            generated = FD.generate_stab_mats(False, C_gen, 0.5 * x[3], 0.5 * x[4], res, V_low, V_high)
        case 2:
            C_gen = CI.Cap(30 * init[0], 30 * init[1], 8 * x[0], 8 * x[1], 8 * x[2])
            generated = FD.generate_stab_mats(False, C_gen, 0.5 * init[5], 0.5 * init[6], res, V_low, V_high)

    diff = generated - measured

    lf = np.sum(np.power(diff, 2.)) / res ** 2

    return lf


def FS_FD(x, measured, res, V_low_L, V_high_L, V_low_R, V_high_R):

    C_gen = CI.Cap(x[0], x[1], x[2], x[3], x[4])

    generated = FD.generate_stab_mats(True, C_gen, x[5], x[6], res, V_low_L, V_high_L, V_low_R, V_high_R)

    ft_m = np.abs(np.fft.fft2(measured))
    ft_g = np.abs(np.fft.fft2(generated))

    ft_m = np.log(np.abs(ft_m/(np.max(ft_m) + 0.0001)) + 0.0001)
    ft_g = np.log(np.abs(ft_g/(np.max(ft_g) + 0.0001)) + 0.0001)

    diff = ft_m - ft_g

    lf = np.sum(np.power(diff, 2.)) / res ** 2

    return lf


def INT_LIN_FD(x, opt_cycle, init, measured, res, V_low, V_high):
    meas = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            meas[i, j] = np.sum(measured[0:i + 1, 0: j + 1])

    match opt_cycle:
        case 0:
            gen = np.zeros((res, res))
            C_gen = CI.Cap(30 * x[0], 30 * x[1], 8 * x[2], 8 * x[3], 8 * x[4])
            generated = FD.generate_stab_mats(True, C_gen, 0.5 * x[5], 0.5 * x[6], res, V_low, V_high)
            for i in range(res):
                for j in range(res):
                    gen[i, j] = np.sum(generated[0:i + 1, 0: j + 1])
        case 1:
            gen = np.zeros((res, res))
            C_gen = CI.Cap(30 * init[0], 30 * init[1], 8 * x[0], 8 * x[1], 8 * x[2])
            generated = FD.generate_stab_mats(True, C_gen, 0.5 * x[3], 0.5 * x[4], res, V_low, V_high)
            for i in range(res):
                for j in range(res):
                    gen[i, j] = np.sum(generated[0:i + 1, 0: j + 1])
        case 2:
            gen = np.zeros((res, res))
            C_gen = CI.Cap(30 * init[0], 30 * init[1], 8 * x[0], 8 * x[1], 8 * x[2])
            generated = FD.generate_stab_mats(True, C_gen, 0.5 * init[5], 0.5 * init[6], res, V_low, V_high)
            for i in range(res):
                for j in range(res):
                    gen[i, j] = np.sum(generated[0:i + 1, 0: j + 1])

    diff = gen - meas

    lf = np.sum(np.power(diff, 2.)) / res ** 2

    return lf

def INT_LIN(x, measured, res, V_lowL, V_highL, V_lowR, V_highR):
    #measured = measured/np.sum(measured)
    meas = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            meas[i, j] = np.sum(measured[0:i + 1, 0: j + 1])

    gen = np.zeros((res, res))
    C_gen = CI.Cap(x[0], x[1], x[2], x[3], x[4])
    generated = FD.generate_stab_mats(True, C_gen, x[5], x[6], res, V_lowL, V_highL, V_lowR, V_highR)
    #if (np.sum(generated) != 0.):
    #    generated = generated/np.sum(generated)


    for i in range(res):
        for j in range(res):
            gen[i, j] = np.sum(generated[0:i + 1, 0: j + 1])

    diff = gen - meas

    lf = np.sum(np.power(diff, 2.)) / res ** 2

    return lf/res**2

def BLUR(x, measured, res, V_low, V_high):


    C_gen = CI.Cap(x[0], x[1], x[2], x[3], x[4])
    generated = FD.generate_stab_mats(True, C_gen, x[5], x[6], res, V_low, V_high)

    k = int(res/20)

    kernel = np.ones((k, k))

    generated = sig.convolve2d(generated, kernel, mode="same")
    measured = sig.convolve2d(measured, kernel, mode="same")

    diff = generated - measured

    lf = np.sum(np.power(diff, 2.)) / res ** 2

    return lf/res**2


def MSE_INT_FD(x, measured, res, V_lowL, V_highL, V_lowR, V_highR):
    C_gen = CI.Cap(30 * x[0], 30 * x[1], 8 * x[2], 8 * x[3], 8 * x[4])
    generated = FD.generate_stab_mats(True, C_gen, 0.5 * x[5], 0.5 * x[6], res, V_lowL, V_highL, V_lowR, V_highR)

    int_measured = np.zeros((res, res))
    int_generated = np.zeros((res, res))

    N_m = 0
    N_g = 0
    j = 0
    for i in range(1, res):
        if (measured[i, j] != 0 and measured[i - 1, j] == 0):
            N_m = N_m + 1

        if (generated[i, j] != 0 and generated[i - 1, j] == 0):
            N_g = N_g + 1

        for j in range(1, res):

            temp_m = N_m
            temp_g = N_g

            if (measured[i, j] != 0 and measured[i, j - 1] == 0):
                temp_m = temp_m + 1

            if (generated[i, j] != 0 and generated[i, j - 1] == 0):
                temp_g = temp_g + 1

            int_measured[i, j] = temp_m
            int_generated[i, j] = temp_g

    diff = int_generated - int_measured

    lf = np.sum(np.power(diff, 2.)) / res ** 2

    return lf


