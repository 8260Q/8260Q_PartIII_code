import numpy as np
import scipy.constants as const

q = const.e / np.power(10., -19)

class Cap:
    def __init__(self, Cg1, Cg2, C_m, CC1, CC2):
        self.CC1 = CC1
        self.CC2 = CC2
        self.C_m = C_m
        self.Cg1 = Cg1
        self.Cg2 = Cg2


def e_config(n):
    N = np.zeros(((n + 1)**2,2))
    for i in range((n + 1)**2):
        N[i, 0] = i//(n + 1)
        N[i, 1] = i%(n + 1)

    return N

def CI_E(n, Vg1, Vg2, C):
    N =  e_config(n)
    CC1 = C.CC1
    CC2 = C.CC2
    C_m = C.C_m
    Cg1 = C.Cg1
    Cg2 = C.Cg2
    C_L = Cg1 + C_m + CC1
    C_R = Cg2 + C_m + CC2

    ECL = q * q * (C_L * C_R) / (C_L * (C_L * C_R - C_m ** 2))
    ECR = q * q * (C_L * C_R) / (C_R * (C_L * C_R - C_m ** 2))
    ECM = q * q * C_m ** 2 / (C_m * (C_L * C_R - C_m ** 2))


    E = np.zeros(np.shape(N)[0])

    for i in range(np.shape(N)[0]):
        N_L = N[i, 0]
        N_R = N[i, 1]

        f1 = -(Cg1*Vg1*(N_L*ECL + N_R*ECM) + Cg2*Vg2*(N_L*ECM + N_R*ECR))/q
        f2 = ((Cg1**2)*(Vg1**2)*ECL/2 + (Cg2**2)*(Vg2**2)*ECR/2 + Cg1*Cg2*Vg1*Vg2*ECM)/q**2
        E[i] = ECL/2*N_L**2 + ECR/2*N_R**2 + N_L*N_R*ECM + f1 + f2

    return N, E
