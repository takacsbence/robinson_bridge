import sys
import numpy as np
from numpy.linalg import inv
import random
import math
import matplotlib.pyplot as plt
import time
from slice import circle_func, RANSAC, tr, inv_tr, load_point_cloud

def plot_section(data, z):
    figure, axes = plt.subplots()
    plt.plot(data[:,0], data[:,1], 'o', markersize=2)
    plt.grid(True)
    axes.set_aspect(1)
    #plt.xlim(-2, 2)
    #plt.ylim(-2, 2)

    plt.xlabel('<- Csepel    x (m)   Stadium->')
    plt.ylabel('<- rotation axis   y (m)   pilon top->')
    plt.title('section at ' + str(z) + ' m')
    plt.savefig('section_' + str(z) + '.png')
    plt.close()
   
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} pcfile")
        sys.exit()

    # load point cloud
    file_path = sys.argv[1]
    pc = load_point_cloud(file_path)
    n = pc.shape[0]

    # transformation parameters
    # shift
    dx = 651521.114
    dy = 235233.065
    dz = 103.132
    # rotation angles
    alfa1 = -9.0    # elevation angle
    alfa2 = 0
    alfa3 = -251.95206 # whole circle bearing of the pilon

    # transformation matrix
    trm = tr(dx, dy, dz, alfa1, alfa2, alfa3)

    # last column as homogenous coordinates
    hom = np.ones((n, 1))
    pc1 = np.append(pc, hom, axis=1)
    # 3D transformation
    pc1 = np.dot(pc1, trm)
    # delete last column
    pc1 = np.delete(pc1, 3, 1)

    dz0 = 0.05  # thickness of the section
    z0 = 4.259 # first section plane, 4.259
    dz = 5.000 # difference between sections, 5.000
    z_max = 65 # last section plane, 65.000
    x_limit = 2.0 # filter out points not on the pilon

    #output file
    fout_name = file_path[:-4] + "_results.txt"
    fout = open(fout_name, "w")
    isplot = False

    z = z0
    while z < z_max:

        #point in the section
        sec = pc1[abs(pc1[:,2] - z) < dz0]

        # remove points far from origin 
        sec = sec[abs(sec[:,0]) < x_limit]
        sec = sec[abs(sec[:,1]) < x_limit]

        nsec = sec.shape[0]
        print(nsec, z)
        #plot_section(sec, z)

        for i in range(1):  # number of repitions
            for j in range(20): # tolerance for in or outlier
                tol = 0.001 + j * 0.001
                for k in range(1, 100, 1):  # nr of iterations

                    # execute ransac algorithm
                    ransac = RANSAC(sec[:,0], sec[:,1], k, tol, isplot, z)
                    xc, yc, R, nin, nout, rms = ransac.execute_ransac()

                    print(f"{z:.3f} {nsec:d} {xc:.3f} {yc:.3f} {R:.3f} {nin:d} {nout:d} {rms:.3f} {k:d} {tol:.3f}", file=fout)

        z += dz

    fout.close()
