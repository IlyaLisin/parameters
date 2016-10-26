import numpy as np
import cv2
from fileLogger import FileLogger

# load the image
image = cv2.imread("images/0000.png")
# define the list of boundaries

# TODO сделать другую бинаризацию

mask = cv2.inRange(image, np.array([0,0,0], dtype="uint8"), np.array([100,100,100], dtype="uint8"))
im = mask

ret,thresh = cv2.threshold(im,127,255,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

im_with_keypoints = image
for i in range(len(contours)):
    im_with_keypoints = cv2.drawContours(im_with_keypoints, contours, i, (255, 0, 0), 2)

# log params
logger = FileLogger()

# CALCULATE PARAMETERS
kernelCount = len(contours)
logger.log("Число зерен", str(kernelCount))

cntSquares = [cv2.contourArea(cnt) for cnt in contours]

minS = min(cntSquares)
maxS = max(cntSquares)
averS = sum(cntSquares) / len(contours)
logger.log("Минимальная площадь зерна", str(minS))
logger.log("Максимальная площадь зерна", str(maxS))
logger.log("Средняя площадь зерна", str(averS))

cntPerimeters = [cv2.arcLength(cnt, True) for cnt in contours]

minP = min(cntPerimeters)
maxP = max(cntPerimeters)
averP = sum(cntPerimeters) / len(contours)
logger.log("Минимальный периметр зерна", str(minP))
logger.log("Максимальный периметр зерна", str(maxP))
logger.log("Средний периметр зерна", str(averP))

#print (kernelCount)
# TODO балл зерна, метод подсчета зерен, ГОСТ-5639-82, 3.4

# Show
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)