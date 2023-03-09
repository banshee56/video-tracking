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
    print('rect', rect)
    # put your implementation here
    
    img = It1
    template = It
    # interpolate both the input images
    x = np.arange(0, img.shape[1])
    y = np.arange(0, img.shape[0])
    template_spline = RectBivariateSpline(x, y, template.T)     # spline of template image
    img_spline = RectBivariateSpline(x, y, img.T)

    # create grid of points on image
    xt, yt = createGrid(x1, y1, x2, y2)                         # creating grid for rect
    xt, yt = xt.ravel(), yt.ravel()
    T = template_spline.ev(xt, yt)                              # getting template points over rect

    jacobian = getJacobian(xt, yt)                              # shape (7200, 2, 6)
    delta_p = np.array([1, 1, 1, 1, 1, 1])                      # starting parameters > threshold to run the loop
    iter = 0                                                    # number of iterations so far
    
    # run until the magnitude of delta_p is greater than the threshold or until we reached maxIters
    while np.linalg.norm(delta_p) >= threshold and iter < maxIters:
        # warp the coordiantes by affine parameters
        M = np.array([[1.0+p[0], p[2],    p[4]],
                      [p[1],    1.0+p[3], p[5]]]).reshape((2, 3))
        
        # create spline for the full warped image
        points = np.stack((xt, yt, np.ones((xt.shape))))
        warped_points = M @ points
        xi = warped_points[0].reshape(xt.shape)
        yi = warped_points[1].reshape(yt.shape)

        I = img_spline.ev(xi, yi)                               # use .ev() to get rect values in warped image

        # get image gradient using .ev() and unroll matrices
        I_x = img_spline.ev(xi, yi, dx=1)                       # x derivative
        I_y = img_spline.ev(xi, yi, dy=1)                       # y derivative

        # reshape
        I_x = np.expand_dims(I_x, 1)                            # changing shape from (7200,) to (7200, 1)
        I_y = np.expand_dims(I_y, 1)
        I_grad = np.stack((I_x, I_y), 2)                        # create gradient matrix, shape (7200, 1, 2)

        # error image
        error_img = (T - I).reshape((T.shape[0], 1, 1))         # shape (7200,)

        # compute delta_p using lstsq
        J = np.matmul(I_grad, jacobian)                         # J, (7200, 1, 2) x (7200, 2, 6) = (7200, 1, 6)

        # compute Hessian
        H = np.transpose(J, (0, 2, 1)) @ J
        H = np.sum(H, 0)

        # setup for lstsq
        b = np.transpose(J, (0, 2, 1)) @ error_img
        b = np.sum(b, 0)

        # compute delta_p
        delta_p = np.linalg.pinv(H) @ b

        # delta_p = np.linalg.lstsq(H, b, rcond=None)[0].reshape((6, 1))          # calculate least squares solution
        p = p + delta_p                                                         # update parameter

        iter += 1

    # reshape the output affine matrix
    M = np.array([[1.0+p[0], p[2],    p[4]],
                      [p[1],    1.0+p[3], p[5]]]).reshape((2, 3))
        

    return M



# helper function to create grid of points between top left and bottom right corners of bounding box
# returns grid of points to be used by RectBivariateSpline.ev()
def createGrid(x1, y1, x2, y2):
    # to get x and y points on grid, can be fractional
    x_range = np.arange(x1, x2+1, 1)
    y_range = np.arange(y1, y2+1, 1)

    # creating points for grid
    xi, yi = np.meshgrid(x_range, y_range)          

    return xi, yi

def getJacobian(xt, yt):
    # create jacobian
    n = 2*xt.shape[0]
    jacobian = np.zeros((n, 6))           # x[0]*x[1] x coordiantes, same for y coordiantes, so x[0]*x[1] length of jacobian for all coordinates

    # jacobian = np.array([[xt, 0, yt, 0, 1, 0],
    #                      [0, xt, 0, yt, 0, 1]])
    jacobian[np.arange(0, n, 2), 0] = xt
    jacobian[np.arange(0, n, 2), 2] = yt
    jacobian[np.arange(0, n, 2), 4] = 1

    jacobian[np.arange(1, n, 2), 1] = xt
    jacobian[np.arange(1, n, 2), 3] = yt
    jacobian[np.arange(1, n, 2), 5] = 1

    return jacobian.reshape(int(n/2), 2, 6)