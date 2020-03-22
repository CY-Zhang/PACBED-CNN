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

def gaussKernel(sigma,imsize):
    x,y = np.meshgrid(range(1,imsize+1),range(1,imsize+1))
    x = x - imsize//2
    y = y - imsize//2
    tmp = -(x**2+y**2)/(2*sigma**2)
    return (1/(2*np.pi*sigma**2))*np.exp(tmp)

def main(path):

    ss_list = [110]

    for ss in range(1):
    
        for slc in range(52):
            base_name = 'depth0_slice'+str(slc)
            base_ext = '_FPavg.npy'
            array_size = (41,41)

            imsize = (np.load(path+base_name+'_X0_Y0'+base_ext)).shape
            out_sz = array_size + imsize
            output = np.zeros(out_sz,dtype=np.float32)
            for x in range(array_size[0]):
                    for y in range(array_size[1]):
                        output[x,y,:,:] = np.load(path+base_name+'_X'+str(x)+'_Y'+str(y)+base_ext)
            output = np.squeeze(output)

            source_size = ss_list[ss]
            px_size = 17.5
            sigma = (source_size/px_size)/(2.355)
            kernel = gaussKernel(sigma,array_size[0])
            fkernel = np.fft.fft2(kernel)
            
            kx,ky = output.shape[2:4]
            #initialize result array
            result = np.zeros(out_sz,dtype=np.float32)
            result = np.squeeze(result)
            for k in range(kx):
                for l in range(ky):
                    #apply convolution for each pixel in (kx,ky) over the whole set of images in (x,y)
                    result[:,:,k,l] = np.fft.fftshift(np.fft.ifft2(fkernel*np.fft.fft2(output[:,:,k,l]))).real
            np.save(path +'STO_' + str(slc) + '_' + str(source_size) + 'pmss.npy',result)
            print('Slice ' + str(slc) + ' finished.')

if __name__ == '__main__':
    Path = sys.argv[1]
    main(Path)
