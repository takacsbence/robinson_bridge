import sys
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

def circle_func(a, b, r, x):
    return (np.sqrt(r**2-(x-a)**2) + b, -np.sqrt(r**2-(x-a)**2) + b)

class RANSAC:
    def __init__(self, x_data, y_data, n, tol, plot, z0):
        self.x_data = x_data
        self.y_data = y_data
        self.n = n
        self.tol = tol
        self.plot = plot
        self.z0 = z0

    def random_sampling(self):
        x = []
        y = []
        save_ran = []
        count = 0

        # get three points from data
        while True:
            ran = np.random.randint(len(self.x_data))

            if ran not in save_ran:
                x.append(self.x_data[ran])
                y.append(self.y_data[ran])
                save_ran.append(ran)
                count += 1

                if count == 3:
                    break

        return np.array(x), np.array(y)

    def make_model(self, x, y):

        b = -(x * x + y * y)    # create pure terms
        ones = np.array([1.0] * np.size(x)) # build coefficient matrix
        a = np.append(x, y)
        a = np.append(a, ones)
        # solve least squares
        par = np.linalg.lstsq(a.reshape(3, np.size(x)).T, b, rcond=None)
        # get original variables
        xc = -0.5 * par[0][0]
        yc = -0.5 * par[0][1]
        R = math.sqrt((par[0][0] ** 2 + par[0][1] ** 2) / 4.0 - par[0][2])
        #print('{:.3f} {:.3f} {:.3f}'.format(xc, yc, R))
        return xc, yc, R

    def eval_model(self, model):

        xc, yc, R = model

        dis = np.sqrt((self.x_data-xc)**2 + (self.y_data-yc)**2) - R
        dis = np.absolute(dis)
        #print(dis)
        return dis

    def execute_ransac(self):
        # find best model
        nmax = 0
        for i in range(self.n):
            x3, y3 = self.random_sampling()
            model = self.make_model(x3, y3)
            dis = self.eval_model(model)
            xk = self.x_data[dis < self.tol]
            yk = self.y_data[dis < self.tol]
            nin = xk.shape[0]

            if nin > nmax:
                xin = xk
                yin = yk
                nmax = nin
                xout = self.x_data[dis >= self.tol]
                yout = self.y_data[dis >= self.tol]
            nout = xout.shape[0]
            nin = nmax

        # regression circle for the best model

        model = self.make_model(xin, yin)
        dis = self.eval_model(model)

        xc, yc, R = model
        dis = np.sqrt((xin-xc)**2 + (yin-yc)**2) - R
        rms = np.sqrt(np.mean(dis**2))

        #print("{:.3f} {:.3f} {:.3f} {:d} {:d} {:.3f}".format(xc, yc, R, nin, nout, rms))
        if self.plot:
            figure, axes = plt.subplots()
            plt.plot(xin, yin, 'o', markersize=2, label='in')
            plt.plot(xout, yout, 'o', markersize=2, label='out')
            plt.plot(xc, yc, 'o', markersize=3, label='center')
            circle1 = plt.Circle((xc, yc), R, fill=False, linewidth=2)
            axes.add_patch(circle1)
            plt.grid(True)
            axes.set_aspect(1)
            #plt.xlim(-2, 2)
            #plt.ylim(-2, 2)
            #plt.legend()
            plt.xlabel('<- Csepel    x (m)   Stadion->')
            plt.ylabel('<- rotation axis   y (m)   top of the pilon->')
            plt.title('section at ' + str(z) + ' m')
            plt.savefig(sys.argv[2] + "/" + str(z) + '.png')
            plt.show(block=False)
            plt.pause(3)
            plt.close()

        return xc, yc, R, nin, nout, rms

def tr(x, y, z, a1, a2, a3):
    """ set up 3D shift and rotation transformation matrix for homogenous coordinates

        Parameters:
        x, y, z - shift parameters along x, y, z axes
        a1, a2, a3 - rotation parameters around x, y, z axes
        returns the transformation matrix
    """
    a1 = np.radians(a1)
    a2 = np.radians(a2)
    a3 = np.radians(a3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [-x, -y, -z, 1]])
    R1 = np.array([[1, 0, 0, 0], [0, np.cos(a1), -np.sin(a1), 0], [0, np.sin(a1), np.cos(a1), 0], [0, 0, 0, 1]])
    R2 = np.array([[np.cos(a2), 0, np.sin(a2), 0], [0, 1, 0, 0], [-np.sin(a2), 0, np.cos(a2), 0], [0, 0, 0, 1]])
    R3 = np.array([[np.cos(a3), -np.sin(a3), 0, 0], [np.sin(a3), np.cos(a3), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    R = np.dot(T, R3)
    R = np.dot(R, R2)
    R = np.dot(R, R1)
    return R

def inv_tr(x, y, z, a1, a2, a3):
    """ set up 3D shift and rotation inverz transformation matrix for homogenous coordinates

        Parameters:
        x, y, z - shift parameters along x, y, z axes
        a1, a2, a3 - rotation parameters around x, y, z axes
        returns the transformation matrix
    """
    a1 = np.radians(a1)
    a2 = np.radians(a2)
    a3 = np.radians(a3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [-x, -y, -z, 1]])
    R1 = np.array([[1, 0, 0, 0], [0, np.cos(a1), -np.sin(a1), 0], [0, np.sin(a1), np.cos(a1), 0], [0, 0, 0, 1]])
    R2 = np.array([[np.cos(a2), 0, np.sin(a2), 0], [0, 1, 0, 0], [-np.sin(a2), 0, np.cos(a2), 0], [0, 0, 0, 1]])
    R3 = np.array([[np.cos(a3), -np.sin(a3), 0, 0], [np.sin(a3), np.cos(a3), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    R = np.dot(R1, R2)
    R = np.dot(R, R3)
    R = np.dot(R, T)
    return R

def load_point_cloud(file_path):
    """ load point cloud from text file
        parameters: file_path - full file name with path
        returns point cloud data as a np array
    """

    # check file exists
    if not Path(file_path).is_file():
        print("input file does not exist")
        exit()

    pc = np.loadtxt(file_path, delimiter=' ', usecols=(0,1,2))
    print(f"{pc.shape[0]} points read")
    return pc

def plot_section(data, z, output_directory):
    """ plot points in a cross section
        parameters: 
            data - point cloud as a np array
            z - distance along the axis, printed in the title of the plot
        returns point cloud data as a np array
    """
    figure, axes = plt.subplots()
    plt.plot(data[:,0], data[:,1], 'o', markersize=2)
    plt.grid(True)
    axes.set_aspect(1)
    #plt.xlim(-2, 2)
    #plt.ylim(-2, 2)

    plt.xlabel('<- Csepel    x (m)   Stadium->')
    plt.ylabel('<- rotation axis   y (m)   pilon top->')
    plt.title('section at ' + str(z) + ' m')
    plt.savefig(output_directory + '/section_' + str(z) + '.png')
    plt.close()

if __name__ == "__main__":

    # check number of command line arguments
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} pcfile output_directory")
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
    #print(trm)

    # last column as homogenous coordinates
    hom = np.ones((n, 1))
    pc1 = np.append(pc, hom, axis=1)
    # 3D transformation
    pc1 = np.dot(pc1, trm)
    # delete last column
    pc1 = np.delete(pc1, 3, 1)

    dz0 = 0.05 # thickness of a section
    z0 = 4.259 # first section plane, 4.259
    dz = 5.000 # difference between sections, 5.000
    z_max = 65 # last section plane, 65.000
    x_limit = 2.0 # filter out points not on the pilon

    # RANSAC parameters
    k = 100 # nr of iterations
    tol = 0.01 # tolerance
    isplot = True

    # create output directory if not exists
    output_directory = sys.argv[2]
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # output file
    fout = open(sys.argv[2] + "/axis.txt", "w")

    ax = [0, 0, 0]
    m = 1
    z = z0
    while z < z_max:

        # points in the section
        sec = pc1[abs(pc1[:,2]-z) < dz0]

        # remove points far from origin
        sec = sec[abs(sec[:,0]) < x_limit]
        sec = sec[abs(sec[:,1]) < x_limit]

        nsec = sec.shape[0]
        plot_section(sec, z, output_directory)

        # execute ransac algorithm
        ransac = RANSAC(sec[:,0], sec[:,1], k, tol, isplot, z)
        xc, yc, R, nin, nout, rms = ransac.execute_ransac()

        print(f"{z:.3f} {nsec:d} {xc:.3f} {yc:.3f} {R:.3f} {nin:d} {nout:d} {rms:.3f}", file=fout)
        #ax0 = [xc, yc, z0]
        #ax = np.append(ax, ax0, axis=0)
        ax = np.vstack([ax, [xc, yc, z]])
        # print section point into a file
        np.savetxt(sys.argv[2] + "/" + str(z) + '.txt', sec, fmt='%.3f')

        z += dz
        m += 1

    # inverse transformation of axis
    # transformation matrix
    trm = inv_tr(-dx, -dy, -dz, -alfa1, -alfa2, -alfa3)
    hom = np.ones((m, 1))
    pc2 = np.append(ax, hom, axis=1)
    # 3D transformation
    pc2 = np.dot(pc2, trm)
    # delete last column
    pc2 = np.delete(pc2, 3, 1)
    # print into a file
    np.savetxt(sys.argv[2] + '/axis_eov.txt', pc2, fmt='%.3f')

    fout.close()

    # plot axis deviation
    plt.plot(1000*ax[:,0], ax[:,2])
    plt.grid(True)
    plt.ylabel('range along the pilon axis (m)')
    plt.xlabel('<- Csepel    dx (mm)   Stadion->')
    plt.title('axis deviation parelell to the bridge')
    plt.savefig(sys.argv[2] + '/xc.png')
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    plt.plot(1000*ax[:,1], ax[:,2])
    plt.grid(True)
    plt.ylabel('range along the pilon axis (m)')
    plt.xlabel('<- rotation axis   dy (mm)   top of the pilon->')
    plt.title('axis deviation perpendicular to the bridge')
    plt.savefig(sys.argv[2] + '/yc.png')
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    plt.plot(ax[:,0], ax[:,1])
    plt.grid(True)
    plt.xlabel('<- Csepel    x (m)   Stadion->')
    plt.ylabel('<- rotation axis   y (m)   top of the pilon->')
    plt.axis('equal')
    #plt.title('cim')
    plt.savefig(sys.argv[2] + '/xc_yc.png')
    plt.show(block=False)
    plt.pause(3)
    plt.close()
