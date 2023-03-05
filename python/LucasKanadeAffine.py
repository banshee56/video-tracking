import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros((6,1))
    x1,y1,x2,y2 = rect

    # put your implementation here
    # the jacobian for translation matrix
    jacobian = np.array([[1, 0],
                         [0, 1]])
    
    # interpolate both the input images
    x = np.arange(0, It.shape[1])
    y = np.arange(0, It.shape[0])
    It_spline = RectBivariateSpline(x, y, It.T)           # spline of template image
    It1_spline = RectBivariateSpline(x, y, It1.T)         # spline of current image

    # create template
    x_range = np.linspace(x1, x2, int(y2-y1))
    y_range = np.linspace(y1, y2, int(x2-x1))
    xt, yt = np.meshgrid(x_range, y_range)                  # coordinates in the window
    T_window = It_spline.ev(xt.flatten(), yt.flatten())     # required window from the template spline
    T = T_window.flatten()                                  # turn into vector

    delta_p = np.array([2, 2])
    iter = 0
    print(np.hypot(delta_p[0], delta_p[1]))
    while iter < maxIters and np.hypot(delta_p[0], delta_p[1]) > threshold:
        # unroll matrices into the right vectors
        x1_w, x2_w, y1_w, y2_w = x1 + p[0], x2 + p[0], y1 + p[1], y2 + p[1]
        x_w_range = np.linspace(x1_w, x2_w, int(y2-y1))
        y_w_range = np.linspace(y1_w, y2_w, int(x2-x1))
        xi, yi = np.meshgrid(x_w_range, y_w_range)      # coordinates in the window
        I_window = It1_spline.ev(xi, yi)

        # image gradient
        I_x = It1_spline.ev(xi, yi, dx=1)               # image gradient over the window
        I_y = It1_spline.ev(xi, yi, dy=1)

        # unroll  matrices
        I = I_window.flatten()                          # translating using shifted indices
        I_x = I_x.flatten()
        I_y = I_y.flatten()
        I_grad = np.stack((I_x, I_y), 1)                # create gradient matrix

        # error image
        b = T - I

        # compute delta_p using lstsq
        H = np.matmul(I_grad, jacobian)                 # Hessian
        delta_p = np.linalg.lstsq(H, b, rcond=None)
        delta_p = delta_p[0]
        p = p + delta_p

        iter += 1


    # reshape the output affine matrix
    M = np.array([[1.0+p[0], p[1],    p[2]],
                 [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)

    return M
