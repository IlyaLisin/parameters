# import sys, cv2 as cv
#
# img = cv.imread(sys.argv[1], 1)
#
# cv.HoughCircles(img, cv.HOUGH_GRADIENT,1,20,
#                             param1=50,param2=30,minRadius=0,maxRadius=0)
#
# cv.imshow('HoughCircles', img)
#
#
#
# cv.waitKey()

#
# import cv2
# import numpy as np
#
# img = cv2.imread('1_1.bmp',0)
# img = cv2.medianBlur(img,5)
# cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#
# circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
#                             param1=50,param2=30,minRadius=0,maxRadius=0)
#
# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     # draw the outer circle
#     cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
#
# cv2.imshow('detected circles',cimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



import numpy as np
import numpy as np
import cv2
im = cv2.imread('1_7.bmp')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[4]
cv2.drawContours(im2, [cnt], 0, (0,255,0), 3)

print(len(contours))

cv2.imshow('detected circles',im2)
cv2.waitKey(0)
cv2.destroyAllWindows()