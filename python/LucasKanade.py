import numpy as np
from scipy.interpolate import RectBivariateSpline

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


    # the jacobian for translation matrix
    jacobian = np.array([[1, 0],
                         [0, 1]])
    
    # interpolate both the input images
    x = np.arange(0, It.shape[1])
    y = np.arange(0, It.shape[0])
    It_spline = RectBivariateSpline(x, y, It.T)                 # spline of template image
    It1_spline = RectBivariateSpline(x, y, It1.T)               # spline of current image

    # create grid of points for template
    x_samples = int(x2-x1)                                      # number of coordinates for range of x values   
    y_samples = int(y2-y1)
    xt, yt = createGrid(x1, y1, x2, y2, x_samples, y_samples)   # creating points for grid

    # create template
    T_window = It_spline.ev(xt, yt)                             # required window from the template spline
    T = T_window.flatten()                                      # turn into vector

    delta_p = np.array([2, 2])                                  # starting parameters > threshold to run the loop
    iter = 0                                                    # number of iterations so far
    while iter < maxIters and np.hypot(delta_p[0], delta_p[1]) > threshold:
        # shift the coordiantes by translation parameters
        shifted_x1, shifted_x2, shifted_y1, shifted_y2 = x1 + p[0], x2 + p[0], y1 + p[1], y2 + p[1]

        # create grid of translated points for the warped image
        xi, yi = createGrid(shifted_x1, shifted_y1, shifted_x2, shifted_y2, x_samples, y_samples)
        I = It1_spline.ev(xi, yi).flatten()                     # use .ev() to get values in warped image

        # get image gradient using .ev() and unroll matrices
        I_x = It1_spline.ev(xi, yi, dx=1).flatten()             # x derivative
        I_y = It1_spline.ev(xi, yi, dy=1).flatten()             # y derivative
        I_grad = np.stack((I_x, I_y), 1)                        # create gradient matrix

        # error image
        b = T - I

        # compute delta_p using lstsq
        H = np.matmul(I_grad, jacobian)                         # Hessian
        delta_p = np.linalg.lstsq(H, b, rcond=None)[0]          # calculate least squares solution
        p = p + delta_p                                         # update parameter

        iter += 1

    return p

# helper function to create grid of points between top left and bottom right corners of bounding box
# returns grid of points to be used by RectBivariateSpline.ev()
def createGrid(x1, y1, x2, y2, x_samples, y_samples):
    # to get x and y points on grid, can be fractional
    x_range = np.linspace(x1, x2, x_samples)
    y_range = np.linspace(y1, y2, y_samples)

    # creating points for grid
    xi, yi = np.meshgrid(x_range, y_range)          

    return xi, yi