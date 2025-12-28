import numpy as np
import xml.etree.ElementTree as ET

def integral(img):
    return np.cumsum(np.cumsum(img.astype(float),axis=0),axis=1)

def rect_sum(ii, x1, y1, x2, y2):
    x1 = max(0,min(x1,ii.shape[1]-1))
    y1 = max(0,min(y1,ii.shape[0]-1))
    x2 = max(0,min(x2,ii.shape[1]-1))
    y2 = max(0,min(y2,ii.shape[0]-1))
    res = ii[y2,x2]
    if y1>0: res -= ii[y1-1,x2]
    if x1>0: res -= ii[y2,x1-1]
    if x1>0 and y1>0: res += ii[y1-1,x1-1]
    return res

def read_gt_box(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bnd = root.find('object').find('bndbox')
    return [
        int(bnd.find('xmin').text),
        int(bnd.find('ymin').text),
        int(bnd.find('xmax').text),
        int(bnd.find('ymax').text)
    ]

def compute_iou(a,b):
    xA = max(a[0],b[0])
    yA = max(a[1],b[1])
    xB = min(a[2],b[2])
    yB = min(a[3],b[3])
    inter = max(0,xB-xA+1)*max(0,yB-yA+1)
    areaA = (a[2]-a[0]+1)*(a[3]-a[1]+1)
    areaB = (b[2]-b[0]+1)*(b[3]-b[1]+1)
    return inter/(areaA+areaB-inter+1e-8)
