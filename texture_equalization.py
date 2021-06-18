import numpy as np
import matplotlib.pyplot as plt
import pyrtools as pt
import matplotlib.image as mpimg
import scipy.ndimage


def equalize(filename):
    img = plt.imread(filename).astype(float)
    filt = 'sp3_filters' # There are 4 orientations for this filter
    pyr = pt.pyramids.SteerablePyramidSpace(img, height=4, order=3)


    '''
    imgList = []
    for s in range(pyr.num_scales):
        band = pyr.pyr_coeffs[(s,0)] #pyr_coeffs is a dictionary
        imgList.append(band)
    pt.imshow(imgList, col_wrap=4)
    '''

    #iterate through all the bands (no highpass for some reason), do the work
    for label in pyr.pyr_coeffs:
        subband = pyr.pyr_coeffs[label]
        #apply the texture equalization algorithm on it
        #convolution
        gaussian1 = scipy.ndimage.gaussian_filter(abs(subband), 5)
        ln = np.log(gaussian1)
        gaussian2 = scipy.ndimage.gaussian_filter(ln, 60)
        sub = ln - gaussian2;
        #normalize
        # c = 0.5 * np.amax(sub)
        # sub = np.divide(c * sub, (c+sub))

        p = np.exp(-1*sub)
        new_subband = np.multiply(subband, p)
        pyr.pyr_coeffs[label] = new_subband
    #reset highpass
    high = pyr.pyr_coeffs['residual_highpass']
    gaussian1 = scipy.ndimage.gaussian_filter(abs(high), 5)
    ln = np.log(gaussian1)
    gaussian2 = scipy.ndimage.gaussian_filter(ln, 60)
    sub = ln - gaussian2;
    p = np.exp(-1*sub)
    new_high = np.multiply(high, p)
    pyr.pyr_coeffs['residual_highpass'] = new_high

    res = pyr.recon_pyr()

    #plot 2 images side by side
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(img)
    f.add_subplot(1,2, 2)
    plt.imshow(res)
    plt.show(block=True)
    

    




