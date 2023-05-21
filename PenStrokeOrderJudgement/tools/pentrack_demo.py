from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob
from PIL import Image, ImageDraw, ImageFont
import numpy

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from PenBall.penball_locate import LocPenBall_v2
from math import sqrt
from CCR.predict import ccr_func
from PenTrack.StrokeExtract import locate_word,stroke_extract,classify_strokes_linear

torch.set_num_threads(1)

parser1 = argparse.ArgumentParser(description='tracking demo')
parser1.add_argument('--config', type=str, help='config file')
parser1.add_argument('--snapshot', type=str, help='model name')
parser1.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser1.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4') or video_name.endswith('mov'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame

def get_vedio_info(video_name):
    cap = cv2.VideoCapture(video_name)

    # 获取视频帧的宽和高
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # 获取视频总帧数和fps
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('w: {}, h: {}, count: {}, fps: {}'.format(w, h, count, fps))

    cap.release()

    return w, h, count, fps

def bbox_dis(last_bbox,bbox):
    # calculate the distance between the last bounding box and the current box
    # bbox: (column, row, width, height)
    last_centerX = last_bbox[0]+(last_bbox[2]/2.0)
    last_centerY = last_bbox[1]+(last_bbox[3]/2.0)
    now_centerX = bbox[0]+(bbox[2]/2.0)
    now_centerY = bbox[1]+(bbox[3]/2.0)

    c_dis = sqrt((last_centerX - now_centerX)**2 + (last_centerY - now_centerY)**2)

    return c_dis

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    ###### draw chinese charater
    if (isinstance(img, numpy.ndarray)): 
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    # front
    fontStyle = ImageFont.truetype(
        "./demo/pen5/STHeiti_Light.ttc", textSize, encoding="utf-8")
    # draw text
    draw.text((left, top), text, textColor, font=fontStyle)
    # turn back to cv2
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

def stroke_judgement(last_frame, penball_traj):
    """"
    word bounding box calculate
    character strock extraction
    strock classification
    word detect
    result compare, output results on frame
    return charac_locate, stroke_text, word, cmp_result, standerd_order
    """
    
    charac_locate = locate_word(penball_traj)
    print(charac_locate)
    ### TODO change to current frame
    # cv2.rectangle(last_frame_copy, (charac_locate[0], charac_locate[1]),
    #                             (charac_locate[2], charac_locate[3]),
    #                             (0, 255, 0), 3)
    
    all_strokes = stroke_extract(penball_traj,last_frame)

    stroke_class = classify_strokes_linear(all_strokes)
    print(stroke_class)

    stroke_matching = {1:"-", 2:"|", 3:"/", 4:"\\", 5:"`"}
    stroke_text = ""
    for i in range(0,len(stroke_class)-1):
        stroke_text += stroke_matching[stroke_class[i]]+", "
    
    stroke_text += stroke_matching[stroke_class[-1]]
    #TODO change to current frame
    # cv2.putText(last_frame_copy, "Detected Stroke Order:"+stroke_text, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # detect the character and get the correct order
    last_frame_roi = last_frame[charac_locate[1]:charac_locate[3],charac_locate[0]:charac_locate[2],:]
    cc_stroke = ccr_func(last_frame_roi)

    #######################
    # result compare
    word = list(cc_stroke.keys())[0]
    standerd_order = list(cc_stroke.values())[0]
    print(word, "\'"+standerd_order+"\'")
    #last_frame_copy = cv2ImgAddText(last_frame_copy, str(word), charac_locate[2]+10, charac_locate[3], (255, 0, 0), 27)
    real_stroke = ""
    for i in stroke_class:
        if i != 5:
            real_stroke += str(i)
        else:
            real_stroke += "4"
    
    print("\'"+real_stroke+"\'")
    cmp_result = None
    if real_stroke == standerd_order:
        cmp_result = True
        # cv2.putText(last_frame_copy, "Order Judgment Result: Correct!", (20,40), \
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cmp_result = False
        # cv2.putText(last_frame_copy, "Order Judgment Result: Wrong!", (20,40), \
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    print(cmp_result)

    return charac_locate, stroke_text, word, cmp_result, standerd_order


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'

    w, h, count, fps = get_vedio_info(args.video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')

    tmp_name = "pen5_traj_new"
    out = cv2.VideoWriter('./demo/output/'+tmp_name+'.mp4', fourcc, fps, (int(w), int(h)), True)
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    
    penball_traj = []       # store the trajectory of penball point
    f=open("./demo/pen5/pen5_penball.txt","w")
    frame_id = 0
    trajdetect_flag = 0      # if start the pen ball detect and store the positions
    last_bbox = None      # restore last bbox location
    stable_num = 0        # how many consecutive frame the bbox have similar place
    nopen_num = 0      # how many consecutive frame do not detect the pen ball
    nopen_frame = []
    charac_loc = None
    for frame in get_frames(args.video_name):
        frame_id = frame_id + 1
        if first_frame:
            try:
                # init_rect = cv2.selectROI(video_name, frame, False, False)
                # print("init_rect: ",init_rect)
                init_rect = (342, 105, 57, 40)  #pen5
            except:
                exit()
            # frame_roi = frame[init_rect[1]:(init_rect[1]+init_rect[3]), init_rect[0]:(init_rect[0]+init_rect[2])]
            # cv2.imwrite("./demo/pen5/pen5_template.png", frame_roi)
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                # if frame_id < 10 or frame_id > 200:
                # cv2.imwrite("./demo/output/pen5/frame"+str(frame_id)+".png", frame)
                bbox = list(map(int, outputs['bbox']))
                
                if trajdetect_flag == 0:
                    # decide if start penball detect
                    if last_bbox is None:
                        last_bbox = bbox
                    else:
                        bdis = bbox_dis(last_bbox,bbox)
                        print("bdis", bdis)
                        if bdis < 5:
                            stable_num = stable_num+1
                        else:
                            stable_num = 0
                        last_bbox = bbox
                    print("stable_num", stable_num)
                    if stable_num > 2:
                        trajdetect_flag = 1
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                (0, 0, 255), 3)
                elif trajdetect_flag == 1:
                    # pen tracking and pen ball detect process
                    print("frameid: ", frame_id, " roi_bbox: ", bbox)
                    penball_point = LocPenBall_v2(frame, bbox)
                    if penball_point is None:
                        print("no pen ball")
                        nopen_num = nopen_num + 1
                        nopen_frame.append((frame_id, frame.copy()))

                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                (0, 255, 0), 3)
                    if len(penball_traj) != 0:
                        for point in penball_traj:
                            cv2.drawMarker(frame, position=point, color=(0, 0, 255),
                                    markerSize =2, markerType=cv2.MARKER_CROSS, thickness=1)
                    if penball_point is not None:
                        print("pen ball point ", penball_point)
                        f.write(str(penball_point[0])+' '+str(penball_point[1])+'\n')
                        penball_traj.append(penball_point)
                        cv2.drawMarker(frame, position=penball_point, color=(255, 0, 0),
                                    markerSize =2, markerType=cv2.MARKER_CROSS, thickness=1)
                        nopen_num = 0
                        nopen_frame = []

                    if nopen_num > 10:
                        # decide if end the tracking
                        trajdetect_flag = 2

                else:
                    # end of the character writing
                    # start to detect the strocks of the charater
                    if charac_loc is None:
                        # choose a middle frame as the frame to extract strocks
                        print("final frame",nopen_frame[6][0])
                        cv2.imwrite("./demo/pen5/pen5_finalpic.png", nopen_frame[6][1])
                        charac_loc, stroke_text, word, cmp_result, standerd_order = stroke_judgement(nopen_frame[6][1], penball_traj)
                    else:
                        cv2.rectangle(frame, (charac_loc[0], charac_loc[1]),
                                (charac_loc[2], charac_loc[3]),
                                (0, 255, 0), 3)
                        cv2.putText(frame, "Detected Stroke Order:"+stroke_text, \
                                    (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        frame = cv2ImgAddText(frame, str(word), \
                                        charac_loc[2]+10, charac_loc[3], (255, 0, 0), 27)
                        if cmp_result:
                            cv2.putText(frame, "Order Judgment Result: Correct!", (20,40), \
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame, "Order Judgment Result: Wrong!", (20,40), \
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                    
        out.write(frame)
        cv2.imshow(video_name, frame)
        cv2.waitKey(2)
        
    f.close()
    out.release()

# def main2():
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
#     cv2.rectangle(last_frame_copy, (charac_locate[0], charac_locate[1]),
#                                 (charac_locate[2], charac_locate[3]),
#                                 (0, 255, 0), 3)
    
#     all_strokes = stroke_extract(penball_traj,last_frame)

#     stroke_class = classify_strokes_linear(all_strokes)
#     print(stroke_class)

#     stroke_matching = {1:"-", 2:"|", 3:"/", 4:"\\", 5:"`"}
#     stroke_text = ""
#     for i in range(0,len(stroke_class)-1):
#         stroke_text += stroke_matching[stroke_class[i]]+", "
    
#     stroke_text += stroke_matching[stroke_class[-1]]
#     #TODO change to current frame
#     cv2.putText(last_frame_copy, "Detected Stroke Order:"+stroke_text, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#     # detect the character and get the correct order
#     last_frame_roi = last_frame[charac_locate[1]:charac_locate[3],charac_locate[0]:charac_locate[2],:]
#     cc_stroke = ccr_func(last_frame_roi)

#     #######################
#     # result compare
#     word = list(cc_stroke.keys())[0]
#     standerd_order = list(cc_stroke.values())[0]
#     print(word, "\'"+standerd_order+"\'")
#     last_frame_copy = cv2ImgAddText(last_frame_copy, str(word), charac_locate[2]+10, charac_locate[3], (255, 0, 0), 27)
#     real_stroke = ""
#     for i in stroke_class:
#         if i != 5:
#             real_stroke += str(i)
#         else:
#             real_stroke += "4"
    
#     print("\'"+real_stroke+"\'")
#     cmp_result = None
#     if real_stroke == standerd_order:
#         cmp_result = True
#         cv2.putText(last_frame_copy, "Order Judgment Result: Correct!", (20,40), \
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#     else:
#         cmp_result = False
#         cv2.putText(last_frame_copy, "Order Judgment Result: Wrong!", (20,40), \
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#     print(cmp_result)
#     cv2.imshow("img_word", last_frame_copy)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.waitKey(1)


if __name__ == '__main__':
    #main2()
    main()

