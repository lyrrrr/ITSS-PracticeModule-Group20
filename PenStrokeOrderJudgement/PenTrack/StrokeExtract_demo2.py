# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from math import sqrt

def calculate_speed(position1, position2, timestamp1, timestamp2):
    distance = np.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)
    time_difference = timestamp2 - timestamp1
    speed = distance / time_difference
    return speed

def is_writing(speed, distance, speed_threshold, distance_threshold):
    return speed < speed_threshold and distance > distance_threshold

def extract_angle_features(stroke_points):
    angles = []
    for i in range(1, len(stroke_points) - 1):
        vector1 = np.array(stroke_points[i-1]) - np.array(stroke_points[i])
        vector2 = np.array(stroke_points[i+1]) - np.array(stroke_points[i])

        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        angles.append(angle)

    return angles

def detect_stroke_type(angles):
    # 简化版，还需升级
    avg_angle = np.mean(angles)
    
    if avg_angle < 45:
        return "横"
    elif avg_angle < 90:
        return "撇"
    elif avg_angle < 135:
        return "竖"
    else:
        return "捺"

def mean_squared_error(image1, image2):
    return np.mean((image1.astype(np.float32) - image2.astype(np.float32))**2)

def angle2(v1,v2):
    # get angle between 2 vectors
    x=np.array(v1)
    y=np.array(v2)

    module_x=np.sqrt(x.dot(x))
    module_y=np.sqrt(y.dot(y))

    if module_x == 0 or module_y == 0:
        return 0.0

    dot_value=x.dot(y)
    if dot_value == 0:
        return 90.0
    
    # get cos value
    cos_theta=dot_value/(module_x*module_y)
    cos_theta = np.clip(cos_theta, -1, 1)
    #print("module_x, module_y, dot_value, cos_theta", module_x, module_y, dot_value, cos_theta)
    angle_radian=np.arccos(cos_theta)

    angle_value=angle_radian*180/np.pi
    return angle_value


#####
# mostly for pen3
# def main():

#     turn_angle_threshold = 110       #pen3 120  #pen4 110
#     gray_threshold = 173      # pen3 160    #pen5 173

#     #last_frame = cv2.imread("./demo/frame270.png")  #pen3
#     last_frame = cv2.imread("./demo/pen5_last2.png")
#     img_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
#     img_copy = last_frame.copy()

#     penball_loclist = []
#     with open('./demo/pen5_penball.txt', 'r') as f:
#         data = f.readlines()  
#         for line in data:
#             point = line.split()      
#             # print(point)
#             # print(type(point))
#             penball_loclist.append([int(point[0]), int(point[1])])

#     ############################
#     # turn angle judgement
#     ############################

#     # loclist_len = len(penball_loclist)
#     # angle_list = []

#     # for i in range(1,loclist_len-1):
#     #     p_now = penball_loclist[i]
#     #     p_last = penball_loclist[i-1]
#     #     p_next = penball_loclist[i+1]

#     #     #print("p_last,p_now,p_next",p_last,p_now,p_next)
#     #     vector1 = np.array(p_now) - np.array(p_last)
#     #     vector2 = np.array(p_next) - np.array(p_now)
#     #     #print("vector1,vector2",vector1,vector2)
#     #     angle_now = angle2(vector1, vector2)
#     #     angle_list.append(angle_now)
    
#     # #print(angle_list)

#     # turnpoints = [penball_loclist[idx] for idx, angle in enumerate(angle_list) if angle > turn_angle_threshold]
#     # print(turnpoints)

#     # for x_t in turnpoints:
#     #     cv2.drawMarker(img_copy,position=x_t,color=(0, 0, 255),
#     #                     markerSize =1, markerType=cv2.MARKER_CROSS, thickness=1)
        
#     # cv2.imshow("img_turn_point", img_copy)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     # cv2.waitKey(1)

#     v_gray_list = []
#     for point in penball_loclist:
#         array_loc = [point[1],point[0]]

#         v_gray = img_gray[array_loc[0]][array_loc[1]]
#         v_gray_list.append(v_gray)

#     #print(v_gray_list)
#     darkpoint = [penball_loclist[idx] for idx, grayv in enumerate(v_gray_list) if grayv <= gray_threshold]

#     darkpoint_uniq = []
    
#     for idx, point in enumerate(darkpoint):
#         if idx == 0:
#             darkpoint_uniq.append(point)
#         else:
#             if point[0]!=darkpoint[idx-1][0] or point[1]!=darkpoint[idx-1][1]:
#                 darkpoint_uniq.append(point)

#     # print(darkpoint)
#     # print("*****************")
#     #print(darkpoint_uniq)
    
#     #print(darkpoint)
#     darkpoint_len = len(darkpoint_uniq)

#     gray_angle_list = []
#     for i in range(1,darkpoint_len-1):
#         p_now = darkpoint_uniq[i]
#         p_last = darkpoint_uniq[i-1]
#         p_next = darkpoint_uniq[i+1]

#         #print("p_last,p_now,p_next",p_last,p_now,p_next)
#         vector1 = np.array(p_now) - np.array(p_last)
#         vector2 = np.array(p_next) - np.array(p_now)
#         #print("vector1,vector2",vector1,vector2)
#         angle_now = angle2(vector1, vector2)
#         gray_angle_list.append(angle_now)
    
#     #print(gray_angle_list)
#     turnpoints = [darkpoint_uniq[idx+1] for idx, angle in enumerate(gray_angle_list) if angle > turn_angle_threshold]
#     #print(turnpoints)

#     # print(darkpoint[0:13])

#     # turnpoints = [[358, 194], [343, 197], [343, 196], [345, 226], [344, 208], [343, 210],
#     #  [344, 210], [343, 211], [344, 210], [343, 210], [360, 211], [332, 213], 
#     #  [333, 229], [319, 231], [371, 233], [371, 232], [371, 233], [394, 198]]
    
#     # # TODO 算turnpoint之间距离
#     # # 建立距离list 如果距离大就放入最终list 如果距离小就一直平均位置 直到遇到距离大的
#     filtered_turnpoints = []
#     filtered_turnpoints.append(turnpoints[0])
#     turnpoints_num = len(turnpoints)
#     #flag = 0
#     for i in range(1,turnpoints_num):
#         #sqrt((x2 - x1)**2 + (y2 - y1)**2) 
#         dis = sqrt((turnpoints[i-1][0]-turnpoints[i][0])**2+(turnpoints[i-1][1]-turnpoints[i][1])**2)
#         if dis < 5.0:
#             # pen3 3.0
#             # filtered_turnpoints[-1][0] = round((filtered_turnpoints[-1][0]+turnpoints[i][0])/2.0)
#             # filtered_turnpoints[-1][1] = round((filtered_turnpoints[-1][1]+turnpoints[i][1])/2.0)
#             pass
#         else:
#             filtered_turnpoints.append(turnpoints[i])
#             #flag = 0
            
#         #turnpoints_dis.append(dis)
    
#     #print("turnpoints_dis",turnpoints_dis)
#     # print(filtered_turnpoints)
#     # print("####### ",darkpoint_uniq.index(filtered_turnpoints[0]))
#     # print("darkpoint_uniq",darkpoint_uniq)


#     ###################################
#     # get point list for every stroke #
#     ###################################
#     # start_idx = 0
#     # end_idx = 0
#     # all_strokes = []

#     # for idx, turnp in enumerate(filtered_turnpoints):
#     #     if idx == 0:
#     #         end_idx = darkpoint_uniq.index(turnp)
#     #         all_strokes.append(darkpoint_uniq[0:end_idx+1])
#     #     elif idx%2 == 1:
#     #         start_idx = darkpoint_uniq.index(turnp)
#     #     elif idx%2 == 0:
#     #         end_idx = darkpoint_uniq.index(turnp)
#     #         all_strokes.append(darkpoint_uniq[start_idx:end_idx+1])

#     #print("all_strokes", all_strokes)
#     # darkpoint_uniq
#     for x_t in darkpoint_uniq:
#         cv2.drawMarker(img_copy,position=x_t,color=(0, 0, 255),
#                         markerSize =1, markerType=cv2.MARKER_CROSS, thickness=1)

#     cv2.imshow("img_gray", img_copy)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.waitKey(1)

    
#######################
#for pen5
#######################
def main():
    turn_angle_threshold = 110       #pen3 120  #pen4 110
    gray_threshold = 173      # pen3 160    #pen5 173

    #last_frame = cv2.imread("./demo/frame270.png")  #pen3
    last_frame = cv2.imread("./demo/pen5_last2.png")
    img_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    img_copy = last_frame.copy()

    penball_loclist = []
    with open('./demo/pen5_penball.txt', 'r') as f:
        data = f.readlines()  
        for line in data:
            point = line.split()      
            # print(point)
            # print(type(point))
            penball_loclist.append([int(point[0]), int(point[1])])

    ############################
    # turn angle judgement
    ############################

    # loclist_len = len(penball_loclist)
    # angle_list = []

    # for i in range(1,loclist_len-1):
    #     p_now = penball_loclist[i]
    #     p_last = penball_loclist[i-1]
    #     p_next = penball_loclist[i+1]

    #     #print("p_last,p_now,p_next",p_last,p_now,p_next)
    #     vector1 = np.array(p_now) - np.array(p_last)
    #     vector2 = np.array(p_next) - np.array(p_now)
    #     #print("vector1,vector2",vector1,vector2)
    #     angle_now = angle2(vector1, vector2)
    #     angle_list.append(angle_now)
    
    # #print(angle_list)

    # turnpoints = [penball_loclist[idx] for idx, angle in enumerate(angle_list) if angle > turn_angle_threshold]
    # print(turnpoints)

    # for x_t in turnpoints:
    #     cv2.drawMarker(img_copy,position=x_t,color=(0, 0, 255),
    #                     markerSize =1, markerType=cv2.MARKER_CROSS, thickness=1)
        
    # cv2.imshow("img_turn_point", img_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    v_gray_list = []
    for point in penball_loclist:
        array_loc = [point[1],point[0]]

        v_gray = img_gray[array_loc[0]][array_loc[1]]
        v_gray_list.append(v_gray)

    #print(v_gray_list)
    darkpoint = [penball_loclist[idx] for idx, grayv in enumerate(v_gray_list) if grayv <= gray_threshold]

    darkpoint_uniq = []
    
    for idx, point in enumerate(darkpoint):
        if idx == 0:
            darkpoint_uniq.append(point)
        else:
            if point[0]!=darkpoint[idx-1][0] or point[1]!=darkpoint[idx-1][1]:
                darkpoint_uniq.append(point)

    # print(darkpoint)
    # print("*****************")
    #print(darkpoint_uniq)
    
    #print(darkpoint)
    darkpoint_len = len(darkpoint_uniq)

    gray_angle_list = []
    for i in range(1,darkpoint_len-1):
        p_now = darkpoint_uniq[i]
        p_last = darkpoint_uniq[i-1]
        p_next = darkpoint_uniq[i+1]

        #print("p_last,p_now,p_next",p_last,p_now,p_next)
        vector1 = np.array(p_now) - np.array(p_last)
        vector2 = np.array(p_next) - np.array(p_now)
        #print("vector1,vector2",vector1,vector2)
        angle_now = angle2(vector1, vector2)
        gray_angle_list.append(angle_now)
    
    #print(gray_angle_list)
    turnpoints = [darkpoint_uniq[idx+1] for idx, angle in enumerate(gray_angle_list) if angle > turn_angle_threshold]
    #print(turnpoints)

    # print(darkpoint[0:13])

    # turnpoints = [[358, 194], [343, 197], [343, 196], [345, 226], [344, 208], [343, 210],
    #  [344, 210], [343, 211], [344, 210], [343, 210], [360, 211], [332, 213], 
    #  [333, 229], [319, 231], [371, 233], [371, 232], [371, 233], [394, 198]]
    
    # # TODO 算turnpoint之间距离
    # # 建立距离list 如果距离大就放入最终list 如果距离小就一直平均位置 直到遇到距离大的
    filtered_turnpoints = []
    filtered_turnpoints.append(turnpoints[0])
    turnpoints_num = len(turnpoints)
    #flag = 0
    for i in range(1,turnpoints_num):
        #sqrt((x2 - x1)**2 + (y2 - y1)**2) 
        dis = sqrt((turnpoints[i-1][0]-turnpoints[i][0])**2+(turnpoints[i-1][1]-turnpoints[i][1])**2)
        if dis < 5.0:
            # pen3 3.0
            # filtered_turnpoints[-1][0] = round((filtered_turnpoints[-1][0]+turnpoints[i][0])/2.0)
            # filtered_turnpoints[-1][1] = round((filtered_turnpoints[-1][1]+turnpoints[i][1])/2.0)
            pass
        else:
            filtered_turnpoints.append(turnpoints[i])
            #flag = 0
            
        #turnpoints_dis.append(dis)
    
    #print("turnpoints_dis",turnpoints_dis)
    # print(filtered_turnpoints)
    # print("####### ",darkpoint_uniq.index(filtered_turnpoints[0]))
    # print("darkpoint_uniq",darkpoint_uniq)

    ###################################
    # get point list for every stroke #
    ###################################
    start_idx = 0
    end_idx = 0
    all_strokes = []

    for idx, turnp in enumerate(filtered_turnpoints):
        if idx == 0:
            end_idx = darkpoint_uniq.index(turnp)
            all_strokes.append(darkpoint_uniq[0:end_idx+1])
        elif idx%2 == 1:
            start_idx = darkpoint_uniq.index(turnp)
        elif idx%2 == 0:
            end_idx = darkpoint_uniq.index(turnp)
            all_strokes.append(darkpoint_uniq[start_idx:end_idx+1])
    
    # test for pen3
    if (len(filtered_turnpoints)-1)%2 == 1:
        ## one point left
        all_strokes.append(darkpoint_uniq[start_idx:])

    print("all_strokes", all_strokes)
    # darkpoint_uniq
    #filtered_turnpoints
    # all_strokes
    for x_t in darkpoint_uniq:
        cv2.drawMarker(img_copy,position=x_t,color=(0, 0, 255),
                        markerSize =2, markerType=cv2.MARKER_CROSS, thickness=1)

    cv2.imshow("img_gray", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

# if __name__ == "__main__":
#     main()








    
