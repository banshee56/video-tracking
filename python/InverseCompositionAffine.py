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
    ### precomputations
    # variables
    # range of values for full image to create splines
    x = np.arange(0, It.shape[1])
    y = np.arange(0, It.shape[0])

    # image spline to compute intensity of warped points
    It1_spline = RectBivariateSpline(x, y, It1.T)
    It_spline = RectBivariateSpline(x, y, It.T)

    # meshgrid results to evaluate rect in spline
    xt, yt = createGrid(x1, y1, x2, y2)
    xt, yt = xt.ravel(), yt.ravel()

    # template gradient
    T = It_spline.ev(xt, yt)
    T_x = It_spline.ev(xt, yt, dx=1)
    T_y = It_spline.ev(xt, yt, dy=1)
    # reshape to get correct shape for T_grad
    T_x = np.expand_dims(T_x, 1)                            # changing shape from (7200,) to (7200, 1)
    T_y = np.expand_dims(T_y, 1)
    T_grad = np.stack((T_x, T_y), 2)       

    # jacobian
    jacobian = getJacobian(xt, yt)

    # hessian
    J = T_grad @ jacobian
    H = np.transpose(J, (0, 2, 1)) @ J
    H = np.sum(H, 0)

    # start loop
    iter = 0                    # iteration
    delta_p_norm = threshold+1  # starting value of delta_p
    M = np.eye(3)
    # creating homogenous points of the original points to warp
    points = np.vstack((xt, yt, np.ones(xt.shape[0])))
    
    while delta_p_norm >= threshold and iter < maxIters:
        # warp the coordiantes by affine parameters
        warped_points = M @ points
        xi, yi = warped_points[0], warped_points[1]
        I = It1_spline.ev(xi, yi)
        
        # compute error image
        err_img = (I - T).reshape(T.shape[0], 1, 1)   

        # least squares solution setup
        b = np.transpose(J, (0, 2, 1)) @ err_img
        b = b.sum(0)

        # compute delta_p and its norm value
        delta_p = np.linalg.pinv(H) @ b
        delta_p_norm = np.linalg.norm(delta_p)

        # delta_p warp
        W = np.array([[1.0 + delta_p[0], delta_p[2], delta_p[4]],
                      [delta_p[1], 1.0 + delta_p[3], delta_p[5]]]).reshape((2, 3))
        W = np.vstack((W, np.array([0, 0, 1])))

        # compute new warp M
        M = np.dot(M, np.linalg.inv(W))

        iter += 1
    
    M = M[0: 2, :]
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