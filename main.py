#!/usr/bin/env python
# coding: utf-8




import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa





def plot_pair(images, gray=False):

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(10,8))
    i=0
    
    for y in range(2):
        if gray:
            axes[y].imshow(images[i], cmap='gray')
        else:
            axes[y].imshow(images[i])
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i+=1
    
    plt.show()





def get_poly(ann_path):
    
    with open(ann_path) as handle:
        data = json.load(handle)
    
    shape_dicts = data['shapes']
    
    return shape_dicts





def create_binary_masks(im, shape_dicts):
    
    blank = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)
    i=0;
    for shape in shape_dicts:
        
        print(i)
  
        points = shape['geometry']['coordinates']
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(blank, [points], 255)
        i=i+1
    return blank


# ### Create Masks for Binary Classification




image_list = sorted(os.listdir('img'), key=lambda x: int(x.split('.')[0]))
annot_list = sorted(os.listdir('anno'), key=lambda x: int(x.split('.')[0]))

for im_fn, ann_fn in zip(image_list, annot_list):
    
    im = cv2.imread(os.path.join('img', im_fn), 0)
    
    ann_path = os.path.join('anno', ann_fn)
    
    shape_dicts = get_poly(ann_path)
    
    im_binary = create_binary_masks(im, shape_dicts)
    
    plt.imsave(im_fn,im_binary)

    plot_pair([im, im_binary], gray=True)
    
    plt.show()
    
