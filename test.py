from scipy.io import loadmat
import scipy.io as sio
import numpy as np
import xml.dom.minidom as minidom
import os
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from matplotlib import pyplot as plt
import copy

def show_boxes(im_path, imid, dets, cls, colors=None):
    """Draw detected bounding boxes."""
    if colors is None:
        colors = ['red' for _ in range(len(dets))]
    im = plt.imread(im_path)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(0, len(dets)):
        bbox = dets[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=colors[i], linewidth=5)
        )
        ax.text(bbox[0]+2, bbox[1] +13,
                '{}'.format(cls[i]),
                bbox=dict(facecolor=colors[i], edgecolor=colors[i],alpha=0.8),
                fontsize=18, color='white')
        plt.axis('off')
        plt.tight_layout()
    # plt.show()
    # image_template = 'test2015_%s.jpg'
    # img = image_template % imid
    # img = image_template % imid.zfill(8)
    dir = '/home/magus/dataset3/hico/'
    plt.savefig(dir + imid)

image_root = '/home/magus/dataset3/hico_20160224_det/images/test2015'
image_template = 'HICO_test2015_%s.jpg'
key=20
vrbs="aaaa"
objs="bbbb"
hbox_det=[340,123,567,456]
obox_det=[140,223,467,456]
num=2
k=4
im_path = os.path.join(image_root,
                       image_template % str(key).zfill(8))

show_boxes(im_path, str(k) + vrbs + str(num), [hbox_det, obox_det], [vrbs, objs], ['red', '#00B0F0'])
