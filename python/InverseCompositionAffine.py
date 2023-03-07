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
    print('rect', rect)
    ###### precomputations
    ### template gradient
    # variables
    template = It
    img = It1
    x = np.arange(0, template.shape[1])
    y = np.arange(0, template.shape[0])
    T_spline = RectBivariateSpline(x, y, template.T)
    img_spline = RectBivariateSpline(x, y, img.T)

    # get template T from the rect portion of image
    xt, yt = createGrid(x1, y1, x2, y2)
    T = T_spline.ev(xt, yt).flatten()

    # template gradient
    T_x = T_spline.ev(xt, yt, dx=1).flatten()
    T_y = T_spline.ev(xt, yt, dy=1).flatten()
    T_grad = np.vstack((T_x, T_y)).reshape((T_x.shape[0], 1, 2))

    # jacobian wrt W(x;0)
    jacobian = getJacobian(xt, yt)

    # Hessian
    J = T_grad @ jacobian
    H = np.transpose(J, (0, 2, 1)) @ J
    H = np.sum(H, 0)                                            # number of iterations so far

    # run until the magnitude of delta_p is greater than the threshold or until we reached maxIters
    delta_p = np.array([1, 1, 1, 1, 1, 1])                      # starting parameters > threshold to run the loop
    iter = 0
    M = np.array([[1.0+p[0], p[2],    p[4]],
                  [p[1],    1.0+p[3], p[5]]]).reshape((2, 3))
    M = np.vstack((M, np.array([0, 0, 1]))).reshape((3, 3))
    
    while np.linalg.norm(delta_p) >= threshold and iter < maxIters:
        # shift the coordiantes by affine parameters
        # create spline for the full warped image
        points = np.stack((xt.flatten(), yt.flatten(), np.ones((xt.ravel().shape))))
        warped_points = M @ points
        xi = warped_points[0].reshape(xt.shape)
        yi = warped_points[1].reshape(yt.shape)

        I = img_spline.ev(xi, yi).flatten()                     # use .ev() to get rect values in warped image

        # error image
        error_img = (I - T).reshape((I.shape[0], 1, 1))         # shape (7200,)

        # compute delta_p using lstsq
        b = np.transpose(J, (0, 2, 1)) @ error_img
        b = np.sum(b, 0)
        print(b.shape)

        delta_p = np.linalg.lstsq(H, b, rcond=None)[0]          # calculate least squares solution

        # update warp 
        # W(x;p) o W(x;delta_p)^{-1} = W(p) * W(delta_p) * x
        W_delta_p = np.array([[1.0+delta_p[0], delta_p[2],    delta_p[4]],
                              [delta_p[1],    1.0+delta_p[3], delta_p[5]]]).reshape((2, 3))
        W_delta_p = np.vstack((W_delta_p, np.array([0, 0, 1])))
        M = M @ np.linalg.inv(W_delta_p)

        iter += 1

    # reshape the output affine matrix
    # M = np.array([[1.0+p[0], p[2],    p[4]],
    #                   [p[1],    1.0+p[3], p[5]]]).reshape((2, 3))
    M = M[0:2, :]
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
