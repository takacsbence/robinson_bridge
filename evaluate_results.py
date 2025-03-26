import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.interpolate

if __name__ == "__main__":

    # check command line arguments
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} result_file output_directory")
        sys.exit()

    # read input file
    fin = sys.argv[1]
    results = pd.read_csv(fin, sep=' ', names=['z', 'nsec', 'xc', 'yc', 'R', 'nin', 'nout', 'rms', 'nr_it', 'tol']) 
    results['tol'] *= 1000
    results['rms'] *= 1000

    # create output directory if not exists
    output_directory = sys.argv[2]
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # planned radius
    x_range = [-1.552, 18.448, 38.448, 55.148, 64.259]
    y_diam = [0.750, 1.000, 1.000, 0.700, 0.200]
    y_interp = scipy.interpolate.interp1d(x_range, y_diam)
    #print(y_interp(9.259))
    
    # plot 3D plots
    colors = ["green", "yellow", "red"]  # Define your colors
    cmap = LinearSegmentedColormap.from_list("GreenYellowRed", colors)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = plt.scatter(results['nr_it'], results['tol'], results['rms'])

    ax.set_xlabel('nr of iteration')
    ax.set_ylabel('tolerance of inoutliers [mm]')
    ax.set_zlabel('RMS [mm]')
    #ax.set_zlim3d(0, 0.025)
    plt.savefig(output_directory + '/3dscatter1.png')
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = plt.scatter(results['nr_it'], results['tol'], c=results['nin'] / results['nsec'], cmap=cmap, alpha=0.8, s=2)
    cbar = plt.colorbar(sc)
    cbar.set_label('inlier ratio')

    ax.set_xlabel('nr of iteration')
    ax.set_ylabel('tolerance for in/outlier [m]')
    ax.set_zlabel('rate of inliers')
    plt.savefig(output_directory + '/3dscatter2.png')
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    # loop over sections
    unique_z_values = results['z'].unique()

    for z in unique_z_values:
        r_planned = y_interp(z)
    
        results2 = results[results['z'] == z]
        #print(results2)
        #sc = plt.scatter(results2['nr_it'], results2['R'], c=results2['tol'], cmap=cmap, alpha=0.8, s=2)
        #cbar = plt.colorbar(sc)
        #cbar.set_label('Tolerance [mm]')

        mean = results2['R'].median()
        mean = r_planned
        rang = 0.05
        bins = np.linspace(mean - rang / 2, mean + rang / 2, 101)

        results3 = results2[abs(results2['R'] - r_planned) < rang / 2]
        print(f"{z:.3f} {r_planned:.3f} {results2.shape[0]} {results3.shape[0]}")

        plt.plot(results2['nr_it'], results2['rms'], '.')
        plt.grid()
        plt.xlabel('number of iteration')
        plt.ylabel('rms [mm]')
        plt.title(f"{z:.3f}m, planned radius = {r_planned:.3f}m")
        #plt.ylim([bins.min(), bins.max()])
        #plt.text(80, -75, fin, style='italic')
        plt.savefig(output_directory + '/itrms_' + str(z) + '.png', dpi=300)
        plt.close()

        plt.plot(results2['nr_it'], results2['R'], '.')
        plt.grid()
        plt.xlabel('number of iteration')
        plt.ylabel('radius [m]')
        plt.title(f"{z:.3f}m")
        plt.ylim([bins.min(), bins.max()])
        #plt.text(80, -75, fin, style='italic')
        plt.savefig(output_directory + '/itR' + str(z) + '.png', dpi=300)
        plt.close()
        #plt.show()

        plt.hist([results2['R']], bins, 
            weights=np.ones(len(results2['R'])) / len(results2['R']))
        plt.grid()
        plt.xlabel('radius [m]')
        plt.ylabel('relative frequency')
        plt.xlim([bins.min(), bins.max()])
        plt.ylim([0, 1])
        plt.title(f"{z:.3f}m, planned radius = {r_planned:.3f}m")
        plt.savefig(output_directory + '/itR_hist_' + str(z) + '.png', dpi=300)
        plt.close()
