import cv2
import numpy as np


def show(name, img, x, y):
    windowStartX = 10
    windowStartY = 50
    windowXoffset = 5
    windowYoffset = 40

    w = img.shape[0] + windowXoffset
    h = img.shape[1] + windowYoffset

    cv2.namedWindow(name)
    cv2.moveWindow(name, windowStartX + w * x, windowStartY + h * y)
    cv2.imshow(name, img)
    # cv2.waitKey(0)


def harrisResponseImage(img):
    ## TODO 1.1
    ## Compute the spatial derivatives in x and y direction.

    scale = 1
    delta = 0
    ddepth = -1

    dIdx = cv2.Sobel(img, ddepth, 1, 0, ksize=3)
    dIdy = cv2.Sobel(img, ddepth, 0, 1, ksize=3)

    show("dI/dx", abs(dIdx), 1, 0)
    show("dI/dy", abs(dIdy), 2, 0)

    ##########################################################
    ## TODO 1.2
    ## Compute Ixx, Iyy, and Ixy with
    ## Ixx = (dI/dx) * (dI/dx),
    ## Iyy = (dI/dy) * (dI/dy),
    ## Ixy = (dI/dx) * (dI/dy).
    ## Note: The multiplication between the images is element-wise (not a matrix
    ## multiplication)!!

    Ixx = np.multiply(dIdx, dIdx)
    Iyy = np.multiply(dIdy, dIdy)
    Ixy = np.multiply(dIdx, dIdy)

    show("Ixx", abs(dIdx), 0, 1)
    show("Iyy", abs(dIdy), 1, 1)
    show("Ixy", abs(dIdx), 2, 1)

    ##########################################################
    ## TODO 1.3
    ## Compute the images A,B, and C by blurring the
    ## images Ixx, Iyy, and Ixy with a
    ## Gaussian filter of size 3x3 and standard deviation of 1.

    kernelSize = (3, 3)
    sdev = 1

    A = cv2.GaussianBlur(Ixx, kernelSize, sdev)
    B = cv2.GaussianBlur(Iyy, kernelSize, sdev)
    C = cv2.GaussianBlur(Ixy, kernelSize, sdev)

    show("A", abs(A) * 5, 0, 1)
    show("B", abs(B) * 5, 1, 1)
    show("C", abs(C) * 5, 2, 1)

    ##########################################################
    ## TODO 1.4
    ## Compute the harris response with the following formula:
    ## R = Det - k * Trace*Trace
    ## Det = A * B - C * C
    ## Trace = A + B
    k = 0.06
    det = np.multiply(A, B) - np.multiply(C, C)
    trace = A + B
    response = det - k * np.square(trace)

    ## Normalize the response image
    dbg = (response - np.min(response)) / (np.max(response) - np.min(response))
    dbg = dbg.astype(np.float32)
    show("Harris Response", dbg, 0, 2)

    ##########################################################
    cv2.imwrite("dIdx.png", (abs(dIdx) * 255.0))
    cv2.imwrite("dIdy.png", (abs(dIdy) * 255.0))

    cv2.imwrite("A.png", (abs(A) * 5 * 255.0))
    cv2.imwrite("B.png", (abs(B) * 5 * 255.0))
    cv2.imwrite("C.png", (abs(C) * 5 * 255.0))

    cv2.imwrite("response.png", np.uint8(dbg * 255.0))

    return response


def harrisKeypoints(response, threshold=0.1):
    ## TODO 2.1
    ## Generate a keypoint for a pixel,
    ## if the response is larger than the threshold
    ## and it is a local maximum.
    ##
    ## Don't generate keypoints at the image border.
    ## Note: Keypoints are stored with (x,y) and images are accessed with (y,x)!!
    points = []

    for y in range(1, response.shape[0] - 1):
        for x in range(1, response.shape[1] - 1):
            if response[y, x] > threshold:
                n1 = response[y + 1, x - 1]
                n2 = response[y + 1, x]
                n3 = response[y + 1, x + 1]

                n4 = response[y, x - 1]
                n5 = response[y, x]
                n6 = response[y, x + 1]

                n7 = response[y - 1, x - 1]
                n8 = response[y - 1, x]
                n9 = response[y - 1, x + 1]

                n = [n1, n2, n3, n4, n5, n6, n7, n8, n9]

                idx_pos = np.argmax(n)

                if idx_pos == 4:
                    points.append(cv2.KeyPoint(x, y, 1))

    return points


def harrisEdges(input, response, edge_threshold=-0.01):
    ## TODO 3.1
    ## Set edge pixels to red.
    ##
    ## A pixel belongs to an edge, if the response is smaller than a threshold
    ## and it is a minimum in x or y direction.
    ##
    ## Don't generate edges at the image border.
    result = input.copy()
    for y in range(1,response.shape[0]-1):
        for x in range(1,response.shape[1]-1):
            if response[y, x] < edge_threshold:

                n1 = response[y + 1, x - 1]
                n2 = response[y + 1, x]
                n3 = response[y + 1, x + 1]

                n4 = response[y, x - 1]
                n5 = response[y, x]
                n6 = response[y, x + 1]

                n7 = response[y - 1, x - 1]
                n8 = response[y - 1, x]
                n9 = response[y - 1, x + 1]

                n = [n1, n2, n3, n4, n5, n6, n7, n8, n9]
                ny = [n2,n5,n8]
                nx = [n4,n5,n6]

                idx_pos = np.argmin(n)
                idx_pos_y = np.argmin(ny)
                idx_pos_x = np.argmin(nx)

                if idx_pos_y == 1 | idx_pos_x == 1:
                    result[y, x] = (0, 0, 255)

    return result


def main():
    input_img = cv2.imread('blox.jpg')  ## read the image
    input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)  ## convert to grayscale
    input_gray = (input_gray - np.min(input_gray)) / (np.max(input_gray) - np.min(input_gray))  ## normalize
    input_gray = input_gray.astype(np.float32)  ## convert to float32 for filtering

    ## Obtain Harris Response, corners and edges
    response = harrisResponseImage(input_gray)
    points = harrisKeypoints(response)
    edges = harrisEdges(input_img, response)

    imgKeypoints1 = cv2.drawKeypoints(input_img, points, outImage=None, color=(0, 255, 0))
    show("Harris Keypoints", imgKeypoints1, 1, 2)
    show("Harris Edges", edges, 2, 2)

    cv2.imwrite("edges.png", edges)
    cv2.imwrite("corners.png", imgKeypoints1)


if __name__ == '__main__':
    main()
