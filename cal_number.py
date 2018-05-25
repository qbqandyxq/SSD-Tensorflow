import os
from collections import Counter
import xml.etree.ElementTree as ET
import math
import random

import numpy as np

xml_path = '/nfshome/xueqin/udalearn/data/VOCdevkit/VOC2007/Annotations'
acc = np.zeros([21])

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

def read_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]

    labels = []
    labels_text = []
    gt_boxes = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))
        bbox = obj.find('bndbox')
        gt_boxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))

    return labels, gt_boxes

file_path = sorted(os.listdir(xml_path))
numbers = np.zeros([21])

for j in range(len(file_path)):#20len(file_path)
    xml_file = os.path.join(xml_path, file_path[j])
    labels, gbox = read_xml(xml_file)
    label = Counter(labels)
    for i in range(1, len(numbers)):
        if label[i]:
            numbers[i] += label[i]
print(numbers)