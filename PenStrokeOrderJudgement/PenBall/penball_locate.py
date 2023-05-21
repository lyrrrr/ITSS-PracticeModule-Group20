#!/usr/bin/env python
import cv2
import numpy as np
from sklearn.cluster import KMeans
import math
from scipy.optimize import leastsq
import time

### the line formula us y*sin(theta)+x*cos(theta) = dist
def error(p,x,y):
    #x,y are array; return value is also array
    return y*math.sin(p[0])+x*math.cos(p[0])-p[1] 

# get the parameters in linear functions
def get_line_func(error, Xi, Yi):
    p0=[0,1]
    Para=leastsq(error,p0,args=(Xi,Yi)) 
    theta = Para[0][0]
    dist = Para[0][1]
    return theta, dist

# get the cross point between 2 lines
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

# locate the pen ball point in the given bbox in the frame
def LocPenBall(frame, bbox):
    # frame: current frame image
    # bbox: (column, row, width, height)
    # return the location of pen ball point [col,row] or None(no pen ball)
    pos_col = bbox[0]
    pos_row = bbox[1]
    w = bbox[2]
    h = bbox[3]

    #vis = True
    vis = False

    img_roi = frame[bbox[1]:(bbox[1]+bbox[3]), bbox[0]:(bbox[0]+bbox[2])]
    if vis:
        cv2.imshow("roi", img_roi)
        cv2.waitKey(0)

    img_roi_copy = img_roi.copy()
    img_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    if vis:
        cv2.imshow("img_gray", img_gray)
        cv2.waitKey(0)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    img_mor = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
    if vis:
        cv2.imshow("img_mor", img_mor)
        cv2.waitKey(0)

    detected_edges = cv2.GaussianBlur(img_mor,(3,3),0)
    detected_edges = cv2.Canny(detected_edges, 90, 270)   # canny detect edge
    if vis:
        cv2.imshow("img_canny", detected_edges)
        cv2.waitKey(0)

    edges = detected_edges.copy()
    # get edges points loc [colums, rows]
    contours, _ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    # change contours to point list [colums, rows]
    edge_pointlist = []
    for contour in contours:
        edge_pointlist.extend(list(contour[:,0]))
    # print("len", len(edge_pointlist))
    if len(edge_pointlist) < 10:
        # no enough point to cluster and get line
        return None

    # get gradients on the edges
    gX = cv2.Sobel(img_mor, cv2.CV_64F, 1, 0)
    gY = cv2.Sobel(img_mor, cv2.CV_64F, 0, 1)
    ori_list = []
    for point in edge_pointlist:
        ori_list.append([gX[point[1]][point[0]], gY[point[1]][point[0]]])
    ori_arr = np.asarray(ori_list)

    # use kmeans clustering to find 2 clusters of gradients (2 edge lines of the pen tip)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(ori_arr)
    class_labels = kmeans.labels_
    print(class_labels)
    # get the points array for 2 lines
    line1 = np.asarray([edge_pointlist[idx] for idx, flag in enumerate(class_labels) if flag==0])
    line2 = np.asarray([edge_pointlist[idx] for idx, flag in enumerate(class_labels) if flag==1])
    line3 = np.asarray([edge_pointlist[idx] for idx, flag in enumerate(class_labels) if flag==2])
    all_lines = [line1, line2, line3]
    lines_len = [np.size(line1,0),np.size(line2,0),np.size(line3,0)]
    pick_line = []

    if vis:
        for point in line1:
            cv2.drawMarker(img_roi_copy,position=point,color=(0, 0, 255),
                            markerSize =1, markerType=cv2.MARKER_CROSS, thickness=1)
        for point in line2:
            cv2.drawMarker(img_roi_copy,position=point,color=(0, 255, 0),
                            markerSize =1, markerType=cv2.MARKER_CROSS, thickness=1)
        for point in line3:
            cv2.drawMarker(img_roi_copy,position=point,color=(0, 50, 100),
                            markerSize =1, markerType=cv2.MARKER_CROSS, thickness=1)
        cv2.imshow("img_edge", img_roi_copy)
        cv2.waitKey(0)

    for idx, linelen in enumerate(lines_len):
        if linelen > 5:
            # if only detect one line or detect noise..
            pick_line.append(idx)

    if len(pick_line) == 0 or len(pick_line) == 1:
        return None
    elif len(pick_line) == 2:
        line1_x =all_lines[pick_line[0]][:,0]
        line1_y =all_lines[pick_line[0]][:,1]
        line2_x =all_lines[pick_line[1]][:,0]
        line2_y =all_lines[pick_line[1]][:,1]

        theta1, dist1 = get_line_func(error, line1_x, line1_y)
        theta2, dist2 = get_line_func(error, line2_x, line2_y)
    elif len(pick_line) == 3:
        line1_x =line1[:,0]
        line1_y =line1[:,1]
        line2_x =line2[:,0]
        line2_y =line2[:,1]
        line3_x =line3[:,0]
        line3_y =line3[:,1]

        # use y*sin(theta)+x*cos(theta) = dist to fit the 2 lines
        ttheta1, tdist1 = get_line_func(error, line1_x, line1_y)
        ttheta2, tdist2 = get_line_func(error, line2_x, line2_y)
        ttheta3, tdist3 = get_line_func(error, line3_x, line3_y)
        # remove the line with k<0
        if -1.0/math.tan(ttheta1) < 0:
            theta1, dist1 = ttheta2, tdist2
            theta2, dist2 = ttheta3, tdist3
        elif -1.0/math.tan(ttheta2) < 0:
            theta1, dist1 = ttheta1, tdist1
            theta2, dist2 = ttheta3, tdist3
        elif -1.0/math.tan(ttheta3) < 0:
            theta1, dist1 = ttheta1, tdist1
            theta2, dist2 = ttheta2, tdist2
        else:
            #all k > 0, pick the longest 2 edges, remove the shortest one
            min_idx = lines_len.index(min(lines_len))
            if min_idx == 0:
                theta1, dist1 = ttheta2, tdist2
                theta2, dist2 = ttheta3, tdist3
            elif min_idx == 1:
                theta1, dist1 = ttheta1, tdist1
                theta2, dist2 = ttheta3, tdist3
            else:
                theta1, dist1 = ttheta1, tdist1
                theta2, dist2 = ttheta2, tdist2
            # theta1, dist1 = ttheta1, tdist1
            # theta2, dist2 = ttheta2, tdist2
            # theta3, dist3 = ttheta3, tdist3

    #draw 2 lines
    if vis:
        test_x =np.array(list(range(1,w-1)))
        test_y1 = dist1/math.sin(theta1)-test_x/math.tan(theta1)
        test_y2 = dist2/math.sin(theta2)-test_x/math.tan(theta2)
        #test_y3 = dist3/math.sin(theta3)-test_x/math.tan(theta3)

        for idx, x_t in enumerate(test_x):
            cv2.drawMarker(img_roi,position=[int(x_t), round(test_y1[idx])],color=(0, 0, 255),
                            markerSize =1, markerType=cv2.MARKER_CROSS, thickness=1)
            
            cv2.drawMarker(img_roi,position=[int(x_t), round(test_y2[idx])],color=(0, 255, 0),
                        markerSize =1, markerType=cv2.MARKER_CROSS, thickness=1)
            
            # cv2.drawMarker(img_roi,position=[int(x_t), round(test_y3[idx])],color=(20, 10, 50),
            #             markerSize =1, markerType=cv2.MARKER_CROSS, thickness=1)

        cv2.imshow("img_line", img_roi)
        cv2.waitKey(0)
    
    # get the cross point location
    point_is_exist, point_loc = cross_point(theta1, dist1, theta2, dist2)
    print("point_is_exist, point_loc", point_is_exist, point_loc)

    if point_is_exist:
        if (-w) < point_loc[0] and point_loc[0] < 2*w \
            and (-h) < point_loc[1] and point_loc[1] < 2*h:
            # -w < x <2w; -h < y <2h
            # position in the whole frame
            return [round(point_loc[0]+pos_col), round(point_loc[1]+pos_row)]
        else:
            # out of range, the 2 edge might not be correctly detected, just skip this frame
            return None
    else:
        # skip this frame, if no cross point
        return None

##################
# locate the pen ball point in the given bbox in the frame
def LocPenBall_v2(frame, bbox):
    # frame: current frame image
    # bbox: (column, row, width, height)
    # return the location of pen ball point [col,row] or None(no pen ball)
    pos_col = bbox[0]
    pos_row = bbox[1]
    w = bbox[2]
    h = bbox[3]

    #vis = True
    vis = False
    img_roi = frame[bbox[1]:(bbox[1]+bbox[3]), bbox[0]:(bbox[0]+bbox[2])]
    img_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    if vis:
        cv2.imshow("img_gray", img_gray)
        cv2.waitKey(0)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    img_mor = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel1)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    img_mor = cv2.morphologyEx(img_mor, cv2.MORPH_CLOSE, kernel1)

    # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # img_mor = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel1)

    if vis:
        cv2.imshow("img_mor", img_mor)
        cv2.waitKey(0)

    #print("np.unique(img_mor) ", np.unique(img_mor))
    _ , img_binary = cv2.threshold(img_mor,130,255,cv2.THRESH_BINARY)
    if vis:
        cv2.imshow("img_binary", img_binary)
        cv2.waitKey(0)

    black_area = np.sum(img_binary == 0)
    black_percent = black_area*(0.1)/(h*w)
    print("black_area,black_percent ",black_area,black_percent)
    # when not detect pen tip
    if black_percent < 0.015:    #pen5   0.018
        return None

    detected_edges = cv2.GaussianBlur(img_mor,(3,3),0)
    detected_edges = cv2.Canny(detected_edges, 90, 270)   # canny detect edge
    # if vis:
    #     cv2.imshow("img_canny", detected_edges)
    #     cv2.waitKey(0)

    edges = detected_edges.copy()
    # edges = cv2.bitwise_not(edges)
    # get edges points loc [colums, rows]
    contours, _ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    # change contours to point list [colums, rows]
    edge_pointlist = []
    for contour in contours:
        edge_pointlist.extend(list(contour[:,0]))
    # print("len", len(edge_pointlist))
    if len(edge_pointlist) < 10:
        # no enough point to cluster and get line
        return None
    
    dist_list = []
    for point in edge_pointlist:
        ### distance to left boundary
        #dist = point[0] - pos_col
        dist = point[0]

        ### distance to point (0,0)
        # dist = point[0]*point[0] + point[1]*point[1]
        dist_list.append(dist)
    
    min_dist_idx = dist_list.index(min(dist_list))

    # pointt = [edge_pointlist[min_dist_idx][0], edge_pointlist[min_dist_idx][1]-1]
    # if vis:
    #     cv2.drawMarker(img_roi, position=pointt, color=(255, 0, 0),
    #                 markerSize =1, markerType=cv2.MARKER_CROSS, thickness=1)
    #     cv2.imshow("img_binary", img_roi )
    #     cv2.waitKey(0)

    return [edge_pointlist[min_dist_idx][0]+pos_col, edge_pointlist[min_dist_idx][1]+pos_row]


# if __name__=="__main__":
#     img_frame = cv2.imread("./PenBall/frame54.png")
#     img_frame_copy = img_frame.copy()
#     #bbox = (340, 115, 26, 38)
#     bbox = [360, 112, 31, 32]
#     start =time.clock()
#     #point_loc = LocPenBall(img_frame, bbox)
#     point_loc = LocPenBall_v2(img_frame, bbox)

#     if point_loc is not None:
#         cv2.drawMarker(img_frame_copy, position=point_loc, color=(255, 0, 0),
#                     markerSize =2, markerType=cv2.MARKER_CROSS, thickness=1)
#         end = time.clock()
#         print('Running time: %s Seconds'%(end-start))
#         cv2.imshow("img_line", img_frame_copy)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         cv2.waitKey(1)
#     else:
#         print("not point is returned")
