import cv2
import numpy as np
from homography import computeHomography


def numInliers(points1, points2, H, threshold):
    inlierCount = 0

    ## TODO 4.1
    ## Compute the number of inliers for the given homography
    ## - Project the image points from image 1 to image 2
    ## - A point is an inlier if the distance between the projected point and
    ##      the point in image 2 is smaller than threshold.
    ## Hint: Construct a Homogeneous point of type 'Vec3' before applying H.

    for i in range(len(points1)):
        pix = points1[i][0]
        piy = points1[i][1]
        qix = points2[i][0]
        qiy = points2[i][1]

        projection_x = (pix, qix)
        projection_y = (piy, qiy)
        value_x = np.abs(qix - H * pix)
        value_y = np.abs(qiy - H * piy)
        if value_x < threshold and value_y < threshold:
            inlierCount += 1


    return inlierCount


def computeHomographyRansac(img1, img2, matches, iterations, threshold):
    points1 = []
    points2 = []
    for i in range(len(matches)):
        points1.append(img1['keypoints'][matches[i].queryIdx].pt)
        points2.append(img2['keypoints'][matches[i].trainIdx].pt)

    ## The best homography and the number of inlier for this H
    bestInlierCount = 0
    for i in range(iterations):
        subset1 = []
        subset2 = []

        ## TODO 4.2
        ## - Construct the subsets by randomly choosing 4 matches.
        ## - Compute the homography for this subset
        ## - Compute the number of inliers
        ## - Keep track of the best homography (use the variables bestH and bestInlierCount)
        x = np.random.randint(0, len(points1) - 1)
        subset1 = points1[x:x + 4]
        subset2 = points2[x:x + 4]

        H = computeHomography(subset1, subset2)
        inliers_count = numInliers(subset1, subset2, H, threshold)


    print("(" + str(img1['id']) + "," + str(img2['id']) + ") found " + str(bestInlierCount) + " RANSAC inliers.")
    return  # bestH
