import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, rect):
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
    # x = np.arange(0, It.shape[0])
    # y = np.arange(0, It.shape[1])
    # T = RectBivariateSpline(x, y, It)
    # I = RectBivariateSpline(x, y, It1)

    # I_x, I_y = np.gradient(It1)

    x = np.arange(0, It.shape[1])
    y = np.arange(0, It.shape[0])
    It_spline = RectBivariateSpline(x, y, It.T)                 # spline of template image
    It1_spline = RectBivariateSpline(x, y, It1.T)               # spline of current image

   # create grid of points on image
    x_samples = int(x2-x1)                                      # number of coordinates for range of x values
    y_samples = int(y2-y1)
    xt, yt = createGrid(x1, y1, x2, y2, x_samples, y_samples)   # creating points for grid

    # create template
    T_window = It_spline.ev(xt, yt)                             # required window from the template spline
    T = T_window.flatten()                                      # turn into vector

    delta_p = np.array([2, 2])                                  # starting parameters > threshold to run the loop
    iter = 0                                                    # number of iterations so far
    # run until the magnitude of delta_p is greater than the threshold or until we reached maxIters
    while np.hypot(delta_p[0], delta_p[1]) > threshold and iter < maxIters:
        # shift the coordiantes by affine parameters
        warped_x1 = (p[0]+1)*x1 + p[1]*y1     + p[2]
        warped_y1 = p[3]*x1     + (p[4]+1)*y1 + p[5]
        warped_x2 = (p[0]+1)*x2 + p[1]*y2     + p[2]
        warped_y2 = p[3]*x2     + (p[4]+1)*y2 + p[5]

        # create grid of translated points for the warped image
        xi, yi = createGrid(warped_x1, warped_y1, warped_x2, warped_y2, x_samples, y_samples)
        I = It1_spline.ev(xi, yi).flatten()                     # use .ev() to get values in warped image

        # get image gradient using .ev() and unroll matrices
        I_x = It1_spline.ev(xi, yi, dx=1).flatten()             # x derivative
        I_y = It1_spline.ev(xi, yi, dy=1).flatten()             # y derivative

        # reshape
        I_x = I_x.reshape((I_x.shape[0], 1))
        I_y = I_y.reshape((I_y.shape[0], 1))
        I_grad = np.stack((I_x, I_y), 2)                        # create gradient matrix

        # error image
        b = T - I

        # compute delta_p using lstsq
        jacobian = getJacobian(xi, yi)
        J = np.matmul(I_grad, jacobian)                         # J
        J = np.reshape(J, (J.shape[0], J.shape[2]))

        delta_p = np.linalg.lstsq(J, b, rcond=None)[0]          # calculate least squares solution
        delta_p = np.reshape(delta_p, (6, 1))
        p = p + delta_p                                         # update parameter

        iter += 1

    # reshape the output affine matrix
    M = np.array([[1.0+p[0], p[1],    p[2]],
                  [p[3],    1.0+p[4], p[5]]]).reshape(2, 3)

    return M



# helper function to create grid of points between top left and bottom right corners of bounding box
# returns grid of points to be used by RectBivariateSpline.ev()
def createGrid(x1, y1, x2, y2, x_samples, y_samples):
    # to get x and y points on grid, can be fractional
    x_range = np.linspace(x1, x2, x_samples)
    y_range = np.linspace(y1, y2, y_samples)

    # creating points for grid
    xi, yi = np.meshgrid(x_range, y_range)          

    return xi, yi



def getJacobian(xt, yt):
    # create jacobian
    n = xt.shape[0]*xt.shape[1]
    jacobian = np.zeros((n, 2, 6))           # x[0]*x[1] x coordiantes, same for y coordiantes, so x[0]*x[1] length of jacobian for all coordinates

    # jacobian = np.array([[xt, 0, yt, 0, 1, 0],
    #                      [0, xt, 0, yt, 0, 1]])
    xt = xt.flatten()
    yt = yt.flatten()
    for i in range(n):
        jacobian[i] = np.array([[xt[i], 0, yt[i], 0, 1, 0],
                                [0, xt[i], 0, yt[i], 0, 1]])

    return jacobian




    # reshape the output affine matrix
    M = np.array([[1.0+p[0], p[1],    p[2]],
                 [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)

    return M
