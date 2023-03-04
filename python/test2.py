import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
from scipy import signal

sigma     = 2

def LucasKanade(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   p: movement vector dx, dy
    
    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(2)     # [tx, ty]       
    x1,y1,x2,y2 = rect
    print(rect)
    # put your implementation here

    # warp 
    warp = np.array([[1, 0, p[0]],
                     [0, 1, p[1]]])
    img_warped = cv2.warpAffine(It1, warp, (1100, 1000))

    # jacobian
    jacobian = np.array([[1, 0],
                         [0, 1]])
    
    # RectBivariateSpline - interpolate
    x = np.arange(0, It.shape[1])
    y = np.arange(0, It.shape[0])
    It_spline = RectBivariateSpline(y, x, It)       # interpolated = spline(indices[0], indices[1], grid=False)
    warped_spline = RectBivariateSpline(y, x, img_warped)

    # gradient
    ### use RectBivariateSpline.ev
    # x1, y1 = spline.ev()

    # unroll matrices into the right vectors
    I = np.ravel(warped_spline(np.arange(x1, x2), np.arange(y1, y2)))      # need to subtract with below, so should be same shape
    T = np.ravel(It_spline(np.arange(x1, x2), np.arange(y1, y2)))              # should just be rectangular portion of It image
    print(I.shape)

    # image gradient
    I_x = warped_spline.ev(x, y, dx=1)
    I_y = warped_spline.ev(x, y, dy=1)

    # sobelx = cv2.Sobel(img_warped, cv2.CV_64F,1,0, ksize=5)
    # sobely = cv2.Sobel(img_warped, cv2.CV_64F,0,1, ksize=5)
    # # img_grad = np.stack((sobelx, sobely), 2)

    # I_x = np.ravel(sobelx[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1])
    # I_y = np.ravel(sobely[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1]) 
    I_grad = np.stack((I_x, I_y), 1)

    # error image
    b = T - I

    # calculate hessian, H
    J = np.matmul(I_grad, jacobian)
    H = np.matmul(np.transpose(J), J)
    delta_p = np.matmul(np.matmul(np.linalg.inv(H), np.transpose(J)), b)
    p = p + delta_p
    # print('delta p shape: '+ str(p.shape))








    # # It is initial guess of the location of the object, so it is the template
    # warp = np.array([[1, 0, p[0]],
    #                  [0, 1, p[1]]])
                         
    # warped_img = cv2.warpAffine(It1, warp, (1000, 1100))    # try numpy roll instead
    # warped_img_rect = warped_img[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1] 

    # template = It[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1]
    # b =  template - warped_img_rect

    # jacobian = np.array([[1, 0],
    #                      [0, 1]])
    
    # sobelx = cv2.Sobel(warped_img, cv2.CV_64F,1,0, ksize=5)
    # sobely = cv2.Sobel(warped_img, cv2.CV_64F,0,1, ksize=5)
    # print('sobx: '+str(sobelx.shape))
    # print('soby: '+str(sobely.shape))

    # img_grad = np.stack((sobelx, sobely), 2)
    # print('grad: '+str(img_grad.shape))


    return p

def calculateImageGrad(img0):
    hsize = 2 * np.ceil(3 * sigma) + 1          # size of kernel
    kernel = signal.gaussian(hsize, std=sigma)  # the 1D gaussian kernel
    h = np.outer(kernel, kernel)                # using outer product to get the 2D kernel
    h = h/h.sum()                               # normalizing the gaussian kernel

    # smoothing the image using convolution
    smoothed = myImageFilter(img0, h)

    # the Sobel filters from class notes
    horizontalSobelFilter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    verticalSobelFilter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # convolution with Sobel filters
    imgx = myImageFilter(smoothed, horizontalSobelFilter)   # image gradient in the x direction
    imgy = myImageFilter(smoothed, verticalSobelFilter)     # image gradient in the y direction

    # calculating gradient direction and magnitude matrices/2d arrays
    gradientDirection = np.rad2deg(np.arctan2(imgy, imgx))
    gradientMagnitude = np.hypot(imgx, imgy)
    return gradientMagnitude

def myImageFilter(img0, h):
    # how much to pad each axis by
    padRLength = h.shape[0]//2      # pad by half the number of rows in h
    padCLength = h.shape[1]//2      # pad by half the number of columns in h
    
    # padding the image with 0's
    padded = np.pad(img0, ((padRLength, padRLength), (padCLength, padCLength)), mode='constant')
    img1 = np.zeros_like(img0)      # the filtered image, initialized to 0s

    # we will go through each value/element in the kernel and calculate the element's contribution to the final image, img1
    # we need the following variables for the calculation of the window for each kernel value
    paddedLen0 = padded.shape[0]    # len of padded image on 1 axis
    paddedLen1 = padded.shape[1]
    hLen0 = h.shape[0]              # len of kernel on 1 axis
    hLen1 = h.shape[1]

    # go though each value in the kernel, h
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            # the value from the kernel to apply to image
            value = h[i, j]

            # the value is multiplied with image intensities in the range [i, valueRange0] horizontally across padded array
            # and range [j, valueRange0] vertically across array
            valueRange0 = paddedLen0 - hLen0 + i
            valueRange1 = paddedLen1 - hLen1 + j

            # the contribution of that value by multiplying with the original (padded) image
            valueContribution = np.multiply(value, padded[i : valueRange0 + 1, j : valueRange1 + 1])

            # add contribution to final image (which is initialized to 0)
            img1 = np.add(img1, valueContribution)
        
    return img1