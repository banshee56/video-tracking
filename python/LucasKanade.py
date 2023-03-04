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
    # warp = np.array([[1, 0, p[0]],
    #                  [0, 1, p[1]]])
    # img_warped = cv2.warpAffine(It1, warp, (1100, 1000))

    # jacobian
    jacobian = np.array([[1, 0],
                         [0, 1]])
    
    # RectBivariateSpline - interpolate
    x = np.arange(0, It.shape[1])
    y = np.arange(0, It.shape[0])
    It_spline = RectBivariateSpline(y, x, It)       # spline of template image
    It1_spline = RectBivariateSpline(y, x, It1)     # spline of current image

    # unroll matrices into the right vectors
    x1_w, x2_w, y1_w, y2_w = x1 + p[0], x2 + p[0], y1 + p[1], y2 + p[1]
    x_w_range = np.arange(x1_w, x2_w+1)
    y_w_range = np.arange(y1_w, y2_w+1)

    I_window = It1_spline(x_w_range, y_w_range)
    T_window = It_spline(np.arange(x1, x2+1), np.arange(y1, y2+1))
    I = np.ndarray.flatten(I_window)                # translating using shifted indices
    T = np.ndarray.flatten(T_window)                # should just be rectangular portion of It image

    # image gradient
    xs, ys = np.meshgrid(x_w_range, y_w_range)      # coordinates in the window
    I_x = It1_spline.ev(xs, ys, dx=1)               # image gradient over the window
    I_y = It1_spline.ev(xs, ys, dy=1)

    # unroll gradient matrices
    I_x = np.ndarray.flatten(I_x)
    I_y = np.ndarray.flatten(I_y)
    I_grad = np.stack((I_x, I_y), 1)

    # error image
    b = T - I

    # delta_p = 0
    # iter = 0
    # while delta_p < threshold or iter > maxIters:
    # calculate hessian, H
    J = np.matmul(I_grad, jacobian)
    H = np.matmul(np.transpose(J), J)
    delta_p = np.matmul(np.matmul(np.linalg.inv(H), np.transpose(J)), b)
    p = p + delta_p

        # iter += 1

    return p
