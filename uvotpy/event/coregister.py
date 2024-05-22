# -*- coding: iso-8859-15 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from stsci.convolve import boxcar

def image_registration(im1, im2):
    img1 = im1.copy()
    img2 = im2.copy()
    b1 = np.median(img1)
    b2 = np.median(img2)
    # set minimum to 0
    img1[im1 < b1] = b1
    img2[im1 < b2] = b2
    # normalise images  
    img1 -= 0.9*b1
    img2 -= 0.9*b2
    norm1 = np.max(im1)
    norm2 = np.max(im2)
    print (f"normalising im1 with {norm1} and im2 with {norm2}")
    img1 = img1/norm1
    img2 = img2/norm2

    # Compute the cross-correlation in the Fourier domain
    cross_correlation = np.fft.fft2(img1) * np.fft.fft2(img2).conj()
    cross_correlation /= np.abs(cross_correlation)
    cross_correlation = np.fft.ifft2(cross_correlation)

    # Find the shift using the peak of the cross-correlation
    shift_values = np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape)
    return shift_values

def with_scipy(file,max_shift=20,skip=True):
    """
    
    PROBLEM: aligns on the MOD8 and the Diffraction pattern from the fit to the distortion
    matrix, but does not align on the features. 
    
    """
    from astropy.io import fits
    from stsci.convolve import boxcar
    # Load your images
    f = fits.open(file)
    n = len(f)
    
    # start at last image with previous one as the refimg, fix last one, add to previous
    imgs = []
    shifts = []
    bkg = []
    for i in np.arange(n-1,0,-1):
    # find median backgrounds (background in clocked grism varies -> clipping part off
        im = f[i].data.copy()
        med = np.median(im) 
        q = im > med 
        im[q] = med
        bkg.append(np.median(im) )
    bkg = np.asarray(bkg)   
    #print (f"background {bkg}")

    for i in np.arange(n-1,1,-1):
        #print (f"processing image number {i} to match previous one {i-1}")
        im_ori_1 = f[i-1].data.copy()
        im_ori_2 = f[i].data.copy()
        # smooth the images with 5x5 boxcar
        im1 = boxcar(im_ori_1,(5,5))
        im2 = boxcar(im_ori_2,(5,5))

        # Perform image registration on the smoothed images
        shift_values = image_registration( im1.copy(), im2.copy() )
        
        # shift the original image for a match 
        im2_shifted = shift(im_ori_2, shift_values)
        
        # stack it onto the previous 
        if i == n-1: 
            #print (f"stacking number {i}(shifted) tp {i-1} ")
            refimg = im_ori_1 + im2_shifted
        else:
            # refimg was consistent with im2 so needs also to be shifted
            refimg_shifted = shift(refimg, shift_values)
            refimg = refimg_shifted + im2_shifted        
        imgs.append(im2_shifted)
        shifts.append(shift_values)
        #print (f"images {i},{i-1} shift = {shift_values} ")
    f.close()
    if skip:
        return refimg
        
    # get the shifts for updating the fits grism image headers order is from last to first
    """ for some reason the imgs do not align properly, so this part of the code is 
        only accessible with parameter skip = False.
    """
    shft2 = []
    for a,b in shifts:
        if abs(a - refimg.shape[0]) < max_shift: a = a-refimg.shape[0]
        if abs(b - refimg.shape[1]) < max_shift: b = b-refimg.shape[1]
        shft2.append([a,b])
    shft2b = shft2.copy()    
    shft2 = np.asarray(shft2)
    for k in np.arange(shft2.shape[0]):
        print (f"shift #{k} {shft2[k]}")
    
    f = fits.open(file,mode='update')
    for k in np.arange(n-1,1,-1):
        shft3 = shft2[n-k-1:,:].sum(0)
        print (f"{k} cumulative shift to 1 = {shft3}")
        f[k].header['CRPIX1']  -= shft3[1] # assuming the shift is for the image peaks 2046
        f[k].header['CRPIX2']  -= shft3[0] # so the correction to the coordinate system 1987
        try:
            f[k].header['CRPIX1S'] -= shft3[1] # assuming the shift is for the image peaks 
            f[k].header['CRPIX2S'] -= shft3[0] # so the correction to the coordinate system 
        except: pass
                                           # is the opposite 
        f[k].header['COMMENT'] = f"shift applied to CRPIX-S of {shft3}"
    f.flush()    # save to f
    f.close()    # close fits file
    return refimg, imgs, shifts, shft2

"""
    for a,b in shfts:
        if abs(a - refimg.shape[0]) < 20: a = a-refimg.shape[0]
        if abs(b - refimg.shape[1]) < 20: b = b-refimg.shape[1]
        shft2.append([a,b])

    for k in range(len(shft2),0,-1):
        print (k)
        shft3[len(shft2)-k-1] = shft2[k-1]

    for k in range(n):
        f[k].header['CRPIX1S'] -= shft3[k][0]
        f[k].header['CRPIX2S'] -= shft3[k][1]
        
    f.flush()
    f.writeto()
     
"""     

"""
# alternative way to align images:

# pip install opencv-python
import cv2
import numpy as np

def match_and_overlay_images(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # FLANN parameters for matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    # FLANN-based matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test as per Lowe's paper
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        print("Not enough good matches to find transformation.")
        return

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find the homography matrix
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Apply the homography to image1
    h, w = image1.shape[:2]
    image1_transformed = cv2.warpPerspective(image1, M, (w, h))

    # Combine the transformed image and image2
    result = cv2.addWeighted(image2, 0.5, image1_transformed, 0.5, 0)

    # Display the result
    cv2.imshow('Overlay', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load images
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# Match and overlay images
match_and_overlay_images(image1, image2)

import cv2
import numpy as np

def match_and_overlay_images(image1, image2):
    # (Same as the previous function)

def process_image_stack(image_stack):
    # Process each pair of consecutive images in the stack
    for i in range(len(image_stack) - 1):
        image1 = image_stack[i]
        image2 = image_stack[i + 1]

        # Match and overlay images
        match_and_overlay_images(image1, image2)

# Load the image stack (assuming you have a list of image file paths)
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg', ...]  # Update with actual file paths
image_stack = [cv2.imread(path) for path in image_paths]

# Process the image stack
process_image_stack(image_stack)

"""