import os
import sys
import scipy.io as sio
import scipy.misc as smisc
from scipy import signal
import numpy as np
import re as regexp
import math
import matplotlib.pyplot as plt
import mrcfile
import scipy
from glob import glob

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def main(folder, target):
	# target at 110pm source size and PEAK=5 for now
	# target = '/srv/home/chenyu/CNN/Data/STO_100nm_PACBED/2_2/'
	PEAK = 5

	# loop for different thickness level
	for slc in range(52):
	    filename = 'STO_' + str(slc) + '_110pmss.npy'
	    data = np.load(folder + filename)
	    # loop for different integration center
	    for ix in range(3):
	        for iy in range(3):
	            # loop for different integration radii
	            for ir in range(11):
	                
	                radius = ir + 1
	                expMask = create_circular_mask(radius*2+1,radius*2+1,center=[radius,radius])
	                px_list = np.nonzero(expMask)
	                x = 18 - 1 + ix
	                y = 18 - 1 + iy
	                # row for non-zero pixels
	                row_list = px_list[0] + y - radius
	                col_list = px_list[1] + x - radius
	                PACBED = np.zeros((160,160))
	                
	                for i in range(len(row_list)):
	                    sim = data[row_list[i],col_list[i],164-80:164+80,164-80:164+80]
	                    sim_noisy = np.random.poisson(sim / np.amax(sim) * PEAK)
	                    PACBED = PACBED + sim_noisy
	                    
	                np.save(target + 'STO_' + str(slc) + '_' + str(ix) + '_' + str(iy) + '_' + str(radius) + '.npy',PACBED)

if __name__ == '__main__':
    folder = sys.argv[1]
    target = sys.argv[2]
    main(folder,target)
