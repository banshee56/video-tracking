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







    return p