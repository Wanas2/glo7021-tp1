from turtle import pos
import numpy as np
from scipy.optimize import fmin
from scipy.spatial import distance
import matplotlib.pyplot as plt



def create_rotation_matrix(theta, ax):
    c, s = np.cos(theta), np.sin(theta)
    if ax == "x":
        return np.array([
            [c, -s, 0],
            [c,  s, 0],
            [0,  0, 1]
        ]) 
    elif ax == "y":
        return np.array([
            [1,  0, 0],
            [0,  c, s],
            [0,  s, c]
        ])
    else:
        return np.array([
            [c, 0, -s],
            [0, 1,  0],
            [s, 0,  c]
        ])


def reprojection(H, focale, L):
    M1 = np.identity(4)
    M1[:3, -1] = -np.array([H[0]/H[3], H[1]/H[3], H[2]/H[3]])

    M2 = np.zeros((3, 4))
    M2[0, 0] = M2[1, 1] = focale
    M2[2, 2] = 1.

    out = []
    for  l in L:
        l_ = np.ones((4, 1))
        l_[:3, 0] = l
        I = M2 @ M1 @ l_
        out.append((float(I[0]/I[2]), float(I[1]/I[2])))
    return np.array(out)


def somme_des_residuels_au_carre(pose_camera, focale, L, C):
    R = create_rotation_matrix(pose_camera[2], "y")
    R_ = np.identity(4)
    R_[:3, :3] = R

    M1 = R_
    M1[0, -1] = pose_camera[0]
    M1[2, -1] = pose_camera[1]

    M2 = np.zeros((3, 4))
    M2[0, 0] = M2[1, 1] = focale
    M2[2, 2] = 1.

    np.seterr(invalid='ignore')

    L_ = []
    for  l in L:
        l_ = np.ones((4, 1))
        l_[:3, 0] = l
        I = M2 @ M1 @ l_
        L_.append((float(I[0]/I[2]), float(I[1]/I[2])))

    return sum([distance.euclidean(C[i], L_[i])**2 for i in range(len(L))])


if __name__ == "__main__":
    L = np.array([(-0.2, 0, 1.2), (0, 0, 1), (0.2, 0, 1.2)])
    focale = 1000

    C = reprojection([0, 0, 0, 1], focale, L)
    print(C)

    pose_solution = fmin(
        somme_des_residuels_au_carre, 
        [0.2, 0.2, 0.2], args=(focale, L, C)
    )

    print(pose_solution)

    plt.show()