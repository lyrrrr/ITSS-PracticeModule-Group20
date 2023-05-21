import os
import torch
from CCR.network import Network
#from CCR.config import config
import pickle
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import random

class_num = 4032
checkpoint = "./CCR/epoch-9-2023-05-02-21-30.pth"
use_gpu = False
cfg_img_size = 64

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def predict(frame_roi):
    setup_seed(20)

    model = Network(class_num).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    model.to(device)
    torch.no_grad()

    # test_images = list()
    # for (root, dirs, filenames) in os.walk(image_dir):
    #     for file in filenames:
    #         filename, ext = os.path.splitext(file)
    #         if not filename.isdigit():
    #             continue
    #         ext = str.lower(ext)
    #         if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
    #             test_images.append(os.path.join(root, file))
    # print(test_images)
    # test_images.sort(key=lambda x: int(os.path.basename(x).split('.')[0])) 

    dict_file = open('./CCR/char_dict', 'rb')
    char_dict = pickle.load(dict_file)

    transformer = transforms.Compose([
        transforms.Resize((cfg_img_size, cfg_img_size)),
        transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.5),
        transforms.GaussianBlur((3, 3)),
        transforms.ToTensor(),
    ])

    results = []

    im = frame_roi
    img = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    #img = Image.open(im)
    img_ = transformer(img).unsqueeze(0)
    img_ = img_.to(device)
    outs = model(img_)

    top3 = torch.topk(outs, 3, dim=-1)[-1].tolist()
    top3 = np.array(top3).flatten()

    result = []
    for i in top3:
        # char = list(char_dict.values())[list(char_dict.values()).index(i)]
        char = list(char_dict.keys())[list(char_dict.values()).index(i)]
        result.append(char)
    # print(result)
    results.append(result)
    return results

def main():
    #print(".....")
    last_frame = cv2.imread("./data/pen5_last2.png")
    #print(last_frame.shape)
    bbox = (351, 82, 454, 198)
    frame_roi = last_frame[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
    frame_roi = cv2.cvtColor(frame_roi,cv2.COLOR_BGR2GRAY)

    _,frame_roi_bi = cv2.threshold(frame_roi,120,255,cv2.THRESH_BINARY)

    # cv2.imshow("OpenCV1",frame_roi_bi)
    # cv2.waitKey()
    frame_roi_new = cv2.cvtColor(frame_roi_bi,cv2.COLOR_GRAY2BGR)

    # clahe  = cv2.createCLAHE()
    # hcl = clahe.apply(frame_roi)
    #hcl = cv2.equalizeHist(frame_roi)

    # print(frame_roi.shape)
    #frame_roi = last_frame
    # cv2.imshow("OpenCV",frame_roi_new)
    # cv2.waitKey()
    results = predict(frame_roi_new)

    chinese_char_map = {}
    with open('./chinese_unicode_table.txt', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines[6:]: 
            line_info = line.strip().split()
            chinese_char_map[line_info[0]] = line_info[8]

    orderList = []
    word_order_map = {}
    
    for i in range(len(results)):
        wordList = results[i]
        word = wordList[0]
        order = chinese_char_map[word]
        orderList.append(order)
        word_order_map[word] = order

    print(orderList)
    print(word_order_map)

def ccr_func(frame_roi):
    # get the character region image
    # return the recognized character and the correct stroke order

    # last_frame = cv2.imread("./data/pen5_last2.png")
    # bbox = (351, 82, 454, 198)
    # frame_roi = last_frame[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
    frame_roi = cv2.cvtColor(frame_roi,cv2.COLOR_BGR2GRAY)
    _,frame_roi_bi = cv2.threshold(frame_roi,120,255,cv2.THRESH_BINARY)
    frame_roi_new = cv2.cvtColor(frame_roi_bi,cv2.COLOR_GRAY2BGR)

    results = predict(frame_roi_new)

    chinese_char_map = {}
    with open('./CCR/chinese_unicode_table.txt', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines[6:]: 
            line_info = line.strip().split()
            chinese_char_map[line_info[0]] = line_info[8]

    orderList = []
    word_order_map = {}
    
    for i in range(len(results)):
        wordList = results[i]
        word = wordList[0]
        order = chinese_char_map[word]
        orderList.append(order)
        word_order_map[word] = order

    # print(orderList)
    # print(word_order_map)
    return word_order_map

# if __name__ == "__main__":
#     #main()
