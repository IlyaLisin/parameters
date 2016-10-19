import numpy as np
import cv2

# load the image
image = cv2.imread("images/0000.png")
# define the list of boundaries

boundaries = [
    ([0, 0, 0], [212, 212, 212]),
]

# loop over the boundaries
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
 #   output = cv2.bitwise_and(image, image, mask=mask)

#    cv2.floodFill(image, mask, 0,0)
 #   im = output
    im = mask
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 20000

    params.filterByColor = True
    params.blobColor = 0
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 30000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0
    params.maxCircularity = 1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0
    params.maxConvexity = 1

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0
    params.maxInertiaRatio = 1

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)


    # show the images
    #cv2.imshow("images", np.hstack([image, output]))
    cv2.waitKey(0)