import numpy as np
import cv2


## Compute a homography matrix from 4 point matches
def computeHomography(points1, points2):
    '''
    Solution with OpenCV calls not allowed
    '''
    assert (len(points1) == 4)
    assert (len(points2) == 4)

    ## 3
    ## Construct the 8x9 matrix A.
    ## Use the formula from the exercise sheet.
    ## Note that every match contributes to exactly two rows of the matrix.

    # A = [[-p1x, -p1y,-1,  0,    0,    0, p1xq1x, py1q1x, q1x],
    #     [ 0,    0,   0, -p1x, -p1y, -1, p1xq1y, py1q1y, q1y],
    #     ...
    #     ]

    A = np.zeros(shape=(8, 9))
    for i in range(4):
        pix = points1[i][0]
        piy = points1[i][1]
        qix = points2[i][0]
        qiy = points2[i][1]
        row1 = [-pix, -piy, -1, 0, 0, 0, pix * qix, piy * qix, qix]
        row2 = [0, 0, 0, -pix, -piy, -1, pix * qiy, piy * qiy, qiy]
        A[i * 2] = row1
        A[i * 2 + 1] = row2

    U, s, V = np.linalg.svd(A, full_matrices=True)
    V = np.transpose(V)

    H = np.zeros((3, 3))
    ## 3
    ## - Extract the homogeneous solution of Ah=0 as the rightmost column vector of V.
    ## - Store the result in H.
    ## - Normalize H
    H = V[:, 8].reshape((3, 3))
    H = H * (1/H[2, 2])

    return H


def testHomography():

    points1 = [(1, 1), (3, 7), (2, -5), (10, 11)]
    points2 = [(25, 156), (51, -83), (-144, 5), (345, 15)]

    H = computeHomography(points1, points2)

    print("Testing Homography...")
    print("Your result:" + str(H))

    Href = np.array([[-151.2372466105457, 36.67990057507507, 130.7447340624461],
                     [-27.31264543681857, 10.22762978292494, 118.0943169422209],
                     [-0.04233528054472634, -0.3101691983762523, 1]])

    print("Reference: " + str(Href))

    error = Href - H
    e = np.linalg.norm(error)
    print("Error: " + str(e))

    if (e < 1e-10):
        print("Test: SUCCESS!")
    else:
        print("Test: FAIL!")
    print("============================")
