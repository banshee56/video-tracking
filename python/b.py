import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, rect):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.eye(3)
    h, w = It.shape
    x1,y1,x2,y2 = rect
    threshold = 0.01875
    maxIters = 100


    It_h, It_w = np.arange(h), np.arange(w)
    temp = RectBivariateSpline(It_w, It_h, It.T)
    img = RectBivariateSpline(It_w, It_h, It1.T)
    

    x_range = np.arange(x1, x2+1, 1)
    y_range = np.arange(y1, y2+1, 1)

    # creating points for grid
    X, Y = [a.ravel() for a in np.meshgrid(x_range, y_range)]
    homoXY = np.vstack((X, Y, np.ones(X.shape[0])))
    
    # dT
    T_x = temp.ev(X, Y, dx=1).ravel()
    T_y = temp.ev(X, Y, dy=1).ravel()
    

    # A = np.array([dTx * X, dTx * Y, dTx, dTy * X, dTy * Y, dTy]).T
    # A_dagger = np.linalg.pinv(A.T @ A) @ A.T

    # reshape to get correct shape for T_grad
    T_grad = np.stack((T_x, T_y), 1)    # (7200, 2)

    jacobian = getJacobian(X, Y)        # (7200, 6)
    J = T_grad @ jacobian               
    H = np.transpose(J, (0, 2, 1)) @ J

    print(H.shape)
    exit()
    A_dagger =  H @ J.transpose((0, 2, 1))


    for _ in range(int(maxIters)):
        # Warp I with W
        warpedXY = M @ homoXY
        err = (img.ev(warpedXY[0], warpedXY[1]) - temp.ev(X, Y)).reshape((X.shape[0], 1, 1))

        # Compute dp
        # dp = A_dagger @ err.reshape(-1,1)
        err = J.transpose((0, 2, 1)) @ err
        err = err.sum(0)

        dp = A_dagger @ err
        print(err.shape)
        print(A_dagger.shape)
        exit()
        
        # dp = np.linalg.lstsq(H, err, rcond=None)[0]


        # Update W(x;p) <-- W(x;p) @ W(x;dp)^-1
        dM = np.array([[1+dp[0],   dp[2], dp[4]], 
                       [  dp[1], 1+dp[3], dp[5]]]).reshape((2,3))
        dM = np.vstack((dM, np.array([0, 0, 1.0])))
        M = M @ np.linalg.inv(dM)

        if np.sum(dp**2) < threshold:
            break
    
    return M[:2, :]


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

    return jacobian
