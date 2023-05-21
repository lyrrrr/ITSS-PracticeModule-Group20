#!/usr/bin/env python
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy import polyfit
import math
from scipy.optimize import leastsq


#############################
# get the sub-image in vedio
#############################
# vedio_path = "/../demo/pen1.mov"
# cur_dir = os.path.dirname(os.path.realpath(__file__))
# print("cur_dir", cur_dir)
# cap = cv2.VideoCapture(cur_dir+vedio_path)
# ret, frame = cap.read()

# if ret:
#     print(frame.shape)
# else:
#     print("not read the vedio")

# # selectROI: (min_x,min_y,w,h)
# bbox = (340, 115, 26, 38)
# frame_roi = frame[bbox[1]:(bbox[1]+bbox[3]), bbox[0]:(bbox[0]+bbox[2])]
# print(frame_roi.shape)
# cv2.imwrite("./frame_test.png", frame_roi)
# cv2.imwrite("./PenBall/frame1.png", frame)

###########################
# find best canny threshold
###########################
# def CannyThreshold(lowThreshold):
#     detected_edges = cv2.GaussianBlur(img_gray,(3,3),0)
#     detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
#     dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
#     cv2.imshow('canny demo',dst)

# lowThreshold = 0
# max_lowThreshold = 100
# ratio = 3
# kernel_size = 3

# img = cv2.imread("./PenBall/frame_test.png")
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow('canny demo')
# cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)
 
# CannyThreshold(0)  # initialization
# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()



### the line formula us y*sin(theta)+x*cos(theta) = dist
def error(p,x,y):
    #x,y are array; return value is also array
    return y*math.sin(p[0])+x*math.cos(p[0])-p[1] 

def get_line_func(error, Xi, Yi):
    p0=[0,1]
    Para=leastsq(error,p0,args=(Xi,Yi)) #把error函数中除了p以外的参数打包到args中
    theta = Para[0][0]
    dist = Para[0][1]
    return theta, dist

def cross_point(theta1, dist1, theta2, dist2):
    # y*sin(theta)+x*cos(theta) = dist
    point_is_exist = False
    x_cross = y_cross = 0

    if theta1 == 0 and theta2 != 0:
        x_cross = dist1
        y_cross = dist1/math.sin(theta1)-x_cross/math.tan(theta1)
        point_is_exist = True
    elif theta2 == 0 and theta1 != 0:
        x_cross = dist2
        y_cross = dist2/math.sin(theta2)-x_cross/math.tan(theta2)
        point_is_exist = True
    elif theta1 == 0 and theta2 == 0:
        pass
    elif math.tan(theta1) != math.tan(theta2):
        k1 = -1.0/math.tan(theta1)
        b1 = dist1/math.sin(theta1)
        k2 = -1.0/math.tan(theta2)
        b2 = dist2/math.sin(theta2)
        x_cross = (b2 - b1) * 1.0 / (k1 - k2)
        y_cross = k1 * x_cross * 1.0 + b1 * 1.0
        point_is_exist = True

    return point_is_exist, [x_cross, y_cross]

    # if k1 is None:
    #     if not k2 is None:
    #         x = x1
    #         y = k2 * x1 + b2
    #         point_is_exist = True
    # elif k2 is None:
    #     x = x3
    #     y = k1 * x3 + b1
    # elif not k2 == k1:
    #     x = (b2 - b1) * 1.0 / (k1 - k2)
    #     y = k1 * x * 1.0 + b1 * 1.0
    #     point_is_exist = True

###################
###################

img = cv2.imread("./PenBall/frame_test.png")
img_frame = cv2.imread("./PenBall/frame1.png")
img_frame_copy = img_frame.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detected_edges = cv2.GaussianBlur(img_gray,(3,3),0)
detected_edges = cv2.Canny(detected_edges, 80, 240)   # canny detect edge
edges = detected_edges.copy()

# get edges
#  [colums, rows]
contours, _ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 

edge_pointlist = []
for point in contours[0]:
    edge_pointlist.append([point[0][0], point[0][1]])
print("edge_pointlist_len", len(edge_pointlist))

# get gradients on the edges
gX = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0)
gY = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1)

ori_list = []
# x_list = []
# y_list = []
for point in edge_pointlist:
    ori_list.append([gX[point[1]][point[0]], gY[point[1]][point[0]]])
    # x_list.append(gX[point[1]][point[0]])
    # y_list.append(gY[point[1]][point[0]])

ori_arr = np.asarray(ori_list)

# use kmeans clustering to find 2 clusters of gradients (2 edge lines of the pen tip)
kmeans = KMeans(n_clusters=2, random_state=0).fit(ori_arr)
#print(kmeans.labels_)

class_labels = kmeans.labels_

# get the points array for 2 lines
line1 = np.asarray([edge_pointlist[idx] for idx, flag in enumerate(class_labels) if flag==1])
line2 = np.asarray([edge_pointlist[idx] for idx, flag in enumerate(class_labels) if flag==0])

line1_x =line1[:,0]
line1_y =line1[:,1]
line2_x =line2[:,0]
line2_y =line2[:,1]

# TODO some frame cannnot fully include 2 edges; or just skip

# use polyfit in numpy to fit the straight lines
# position [colums, rows]
# coeff_line1 = polyfit(line1_x, line1_y, 1)
# coeff_line2 = polyfit(line2_x, line2_y, 1)
# print("coeff1", coeff_line1)
# print("coeff2", coeff_line2)

# use y*sin(theta)+x*cos(theta) = dist to fit the 2 lines
theta1, dist1 = get_line_func(error, line1_x, line1_y)
theta2, dist2 = get_line_func(error, line2_x, line2_y)
print("line1:", theta1, dist1)
print("line2:", theta2, dist2)

# get the cross point location
point_is_exist, point_loc = cross_point(theta1, dist1, theta2, dist2)
print(point_is_exist, point_loc)
if point_is_exist:
    # 340, 115
    cv2.drawMarker(img_frame_copy,position=[round(point_loc[0]+340), round(point_loc[1]+115)],color=(255, 0, 0),
                    markerSize =2, markerType=cv2.MARKER_CROSS, thickness=1)
else:
    # TODO skip this frame, if no cross point
    pass

# test_x =np.array(list(range(1,25)))
# # test_y1 = coeff_line1[0]*test_x + coeff_line1[1]
# # test_y2 = coeff_line2[0]*test_x + coeff_line2[1]
# test_y1 = dist1/math.sin(theta1)-test_x/math.tan(theta1)
# test_y2 = dist2/math.sin(theta2)-test_x/math.tan(theta2)

# for idx, x_t in enumerate(test_x):
#     cv2.drawMarker(img_copy,position=[int(x_t), round(test_y1[idx])],color=(0, 0, 255),
#                     markerSize =1, markerType=cv2.MARKER_CROSS, thickness=1)
    
#     cv2.drawMarker(img_copy,position=[int(x_t), round(test_y2[idx])],color=(0, 255, 0),
#                 markerSize =1, markerType=cv2.MARKER_CROSS, thickness=1)

cv2.imshow("img_line", img_frame_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)


######################################
# draw the scatter map of orientations 
# print(len(x_list))
# print(len(y_list))
# plt.figure(figsize=(10, 10))
# plt.scatter(x_list, y_list)
# plt.show()

# point_list = [(5, 10), (6, 11), (6, 12)]
# # (6, 11), (6, 12)
# for point in point_list:
#     # cv2.circle(img_copy, point, 1, (0, 0, 255), 0)
#     cv2.drawMarker(img_copy,position=point,color=(0, 0, 255),
#                    markerSize =1, markerType=cv2.MARKER_CROSS, thickness=1)
 
# print (type(contours))
# print (type(contours[0]))
# print (len(contours))

# print (type(hierarchy))  
# print (hierarchy.ndim)  
# print (hierarchy[0].ndim)  
# print (hierarchy.shape) 

#cv2.drawContours(img_copy,contours,-1,(0,0,255),3)  

######################
# houghlinep not work
######################

# minLineLength = 50
# maxLineGap = 10
# lines = cv2.HoughLinesP(detected_edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
# print("lines: ", lines)
# print("line number: ", len(lines))

# for x1, y1, x2, y2 in lines[0]:
#     cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

# for x1, y1, x2, y2 in lines[1]:
#     cv2.line(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)

# for i, point in enumerate(edge_pointlist):
#     if class_labels[i] == 1:
#         x_line1 = 
#         cv2.drawMarker(img_copy,position=point,color=(0, 0, 255),
#                     markerSize =1, markerType=cv2.MARKER_CROSS, thickness=1)
#     elif class_labels[i] == 0:
#         cv2.drawMarker(img_copy,position=point,color=(0, 255, 0),
#                     markerSize =1, markerType=cv2.MARKER_CROSS, thickness=1)
