import cv2
import numpy as np
import math
from math import sqrt
# from CCR.predict import ccr_func

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_angle(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def has_turning_point(points):
    sigma = 1.05
    for i in range(len(points) - 2):
        B, C, D = points[i], points[i+1], points[i+2]
        BC = get_distance(B, C)
        CD = get_distance(C, D)
        BD = get_distance(B, D)
        if (BC + CD) > BD * sigma:
            return True
    return False

def classify_strokes_linear(stroke_data):
    stroke_types = []  # 用于存储每个笔画的类型
    for stroke in stroke_data:
        # 获取笔画的起点和终点
        start = stroke[0]
        end = stroke[-1]

        # 计算笔画的长度
        length = get_distance(start, end)

        # 计算起点和终点之间的连线的斜率
        if end[0] - start[0] != 0:  # 避免除数为0
            slope = (start[1] - end[1]) / (end[0] - start[0])
        else:
            slope = float('inf')  # 如果除数为0，斜率为无穷大
        
        print("stroke",slope)

        # 判断笔画类型
        if abs(slope) < 0.3:
            # 斜率接近0，判断为横
            stroke_types.append(1)
        elif abs(slope) > 6 or slope == float('inf'):
            # 斜率为无穷大，判断为竖
            stroke_types.append(2)
        elif slope > 0 and 0.3 <= abs(slope) <= 6:
            # 斜率在阈值之间，判断为撇  ## <= 3.6
            stroke_types.append(3)
        elif slope < 0:
            #print(length)
            # 斜率小于0
            if length > 10:  # 假设长度大于5判断为捺，这个阈值可以根据实际情况调整
                stroke_types.append(4)
            else:
                stroke_types.append(5)
        else:
            stroke_types.append(0)  # 未知笔画类型

    return stroke_types

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

def locate_word(penball_traj):
    col = [item[0] for item in penball_traj]
    row = [item[1] for item in penball_traj]
    col_min = min(col)
    col_max = max(col)
    row_min = min(row)
    row_max = max(row)
    
    return (col_min-20, row_min-20, col_max+20, row_max+20)

def stroke_extract(penball_loclist,last_frame):
    turn_angle_threshold = 110      
    gray_threshold = 173      
    
    img_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    img_copy = last_frame.copy()

    v_gray_list = []
    for point in penball_loclist:
        array_loc = [point[1],point[0]]

        v_gray = img_gray[array_loc[0]][array_loc[1]]
        v_gray_list.append(v_gray)

    #### get all the penball point on the written lines
    darkpoint = [penball_loclist[idx] for idx, grayv in enumerate(v_gray_list) if grayv <= gray_threshold]

    #### get unique penball points
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

    ##### get all the turn points from the points on the line
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


    filtered_turnpoints = []
    filtered_turnpoints.append(turnpoints[0])
    turnpoints_num = len(turnpoints)
    #flag = 0
    for i in range(1,turnpoints_num):
        #sqrt((x2 - x1)**2 + (y2 - y1)**2) 
        dis = sqrt((turnpoints[i-1][0]-turnpoints[i][0])**2+(turnpoints[i-1][1]-turnpoints[i][1])**2)
        if dis < 5.0:
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


    #print("all_strokes", len(all_strokes))

    # darkpoint_uniq
    #filtered_turnpoints
    # all_strokes

    # for x_t in all_strokes[3]:
    #     cv2.drawMarker(img_copy,position=x_t,color=(0, 0, 255),
    #                     markerSize =2, markerType=cv2.MARKER_CROSS, thickness=1)

    # cv2.imshow("img_gray", img_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    return all_strokes

# if __name__ == "__main__":

#     last_frame = cv2.imread("./demo/pen5/pen5_last2.png")
#     last_frame_copy = last_frame.copy()
#     penball_traj = []
#     with open('./demo/pen5/pen5_penball.txt', 'r') as f:
#         data = f.readlines()  
#         for line in data:
#             point = line.split()      
#             # print(point)
#             # print(type(point))
#             penball_traj.append([int(point[0]), int(point[1])])
    
#     charac_locate = locate_word(penball_traj)
#     print(charac_locate)
#     ### TODO change to current frame
#     # cv2.rectangle(last_frame_copy, (charac_locate[0], charac_locate[1]),
#     #                             (charac_locate[2], charac_locate[3]),
#     #                             (0, 255, 0), 3)
    
#     all_strokes = stroke_extract(penball_traj,last_frame)

#     stroke_class = classify_strokes_linear(all_strokes)

#     stroke_matching = {1:"-", 2:"|", 3:"/", 4:"\\", 5:"`"}
#     stroke_text = ""
#     for i in range(0,len(stroke_class)-1):
#         stroke_text += stroke_matching[stroke_class[i]]+", "
    
#     stroke_text += stroke_matching[stroke_class[-1]]
#     #TODO change to current frame
#     #cv2.putText(last_frame_copy, "Detected Stroke Order:"+stroke_text, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#     # # detect the character and get the correct order
#     # last_frame_roi = last_frame[charac_locate[1]:charac_locate[3],charac_locate[0]:charac_locate[2],:]
#     # cc_stroke = ccr_func(last_frame_roi)

#     # #######################
#     # # result compare
#     # word = cc_stroke.keys()[0]
#     # standerd_order = cc_stroke.values()[0]
#     # print(word, standerd_order)
#     # real_stroke = ""
#     # for i in stroke_class:
#     #     if i != 5:
#     #         real_stroke += i
#     #     else:
#     #         real_stroke += "4"
    
#     # print(real_stroke)
#     # cmp_result = None
#     # if real_stroke == cc_stroke:
#     #     cmp_result = True
#     # else:
#     #     cmp_result = False

#     # print(cmp_result)
#     # cv2.imshow("img_word", last_frame_copy)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     # cv2.waitKey(1)
