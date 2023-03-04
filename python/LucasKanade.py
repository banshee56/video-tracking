import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

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
    I = np.ravel(warped_spline(np.arange(x1, x2+1), np.arange(y1, y2+1)))      # need to subtract with below, so should be same shape
    T = np.ravel(It_spline(np.arange(x1, x2+1), np.arange(y1, y2+1)))              # should just be rectangular portion of It image

    # image gradient
    xs, ys = np.meshgrid(x, y)
    I_x = warped_spline.ev(xs, ys, dx=1)    # image gradient, not spline
    I_y = warped_spline.ev(xs, ys, dy=1)

    I_x_spline = RectBivariateSpline(y, x, I_x)
    I_y_spline = RectBivariateSpline(y, x, I_y)

    I_x = I_x_spline(np.arange(x1, x2+1), np.arange(y1, y2+1))  # get the points of interest
    I_y = I_y_spline(np.arange(x1, x2+1), np.arange(y1, y2+1))

    # unroll gradients
    I_x = np.ravel(I_x)
    I_y = np.ravel(I_y)
    I_grad = np.stack((I_x, I_y), 1)

    # error image
    b = T - I

    # calculate hessian, H
    J = np.matmul(I_grad, jacobian)
    H = np.matmul(np.transpose(J), J)
    delta_p = np.matmul(np.matmul(np.linalg.inv(H), np.transpose(J)), b)
    p = p + delta_p

    return p
