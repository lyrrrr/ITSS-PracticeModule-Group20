import os
import torch
from crnn.crnn import Network
from config import config
import pickle
from torchvision import transforms
from PIL import Image
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def predict(image_dir):
    setup_seed(20)

    model = Network(config.class_num).eval()
    model.load_state_dict(torch.load(config.checkpoint, map_location='cpu'))
    device = torch.device('cuda:0' if config.use_gpu else 'cpu')
    model.to(device)
    torch.no_grad()

    test_images = list()
    for (root, dirs, filenames) in os.walk(image_dir):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            if not filename.isdigit():
                continue
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                test_images.append(os.path.join(root, file))
    print(test_images)
    test_images.sort(key=lambda x: int(os.path.basename(x).split('.')[0])) 

    dict_file = open('char_dict', 'rb')
    char_dict = pickle.load(dict_file)

    transformer = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.5),
        transforms.GaussianBlur((3, 3)),
        transforms.ToTensor(),
    ])

    results = []
    for im in test_images:
        img = Image.open(im)
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

if __name__ == '__main__':
    results = predict('./data/test_data/4')
#     print(results)

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
