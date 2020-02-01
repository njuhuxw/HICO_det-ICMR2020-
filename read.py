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
import sys

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


# <<<< obsolete
def iou(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    xi1 = max(x11, x21)
    yi1 = max(y11, y21)
    xi2 = min(x12, x22)
    yi2 = min(y12, y22)

    wi = xi2 - xi1 + 1
    hi = yi2 - yi1 + 1

    if wi > 0 and hi > 0:
        areai = wi * hi

        w1 = x12 - x11 + 1
        h1 = y12 - y11 + 1
        area1 = w1 * h1

        w2 = x22 - x21 + 1
        h2 = y22 - y21 + 1
        area2 = w2 * h2

        iou = areai * 1.0 / (area1 + area2 - areai)
    else:
        iou = 0

    return iou


def str_opera(str):
    str_list = str.split("_")
    result = " ".join(str_list)
    return result


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
    dir = '/home/magus/dataset3/hico2/'
    plt.savefig(dir + imid)


class hoi_class:
    def __init__(self, object_name, verb_name, hoi_id):
        self._object_name = object_name
        self._verb_name = verb_name
        self._hoi_id = hoi_id

    def object_name(self):
        return self._object_name

    def verb_name(self):
        return self._verb_name

    def hoi_name(self):
        return self._verb_name + ' ' + self._object_name


def load_hoi_classes(mat_hoi_classes):
    hoi_cls_list = []
    object_cls_list = []
    verb_cls_list = []

    # anno_db = sio.loadmat(os.path.join(self._data_path, 'anno_%s.mat' % self._version))
    # mat_hoi_classes = anno_db['list_action']
    for hoi_cls_id, hoi_cls in enumerate(mat_hoi_classes[:, 0]):
        object_cls_name = hoi_cls['nname'][0]
        if object_cls_name not in object_cls_list:
            object_cls_list.append(object_cls_name)
        verb_cls_name = hoi_cls['vname'][0]
        if verb_cls_name not in verb_cls_list:
            verb_cls_list.append(verb_cls_name)
        hoi_cls_list.append(hoi_class(object_cls_name, verb_cls_name, hoi_cls_id))

    return hoi_cls_list, object_cls_list, verb_cls_list

def load_all_annotations(mat_anno_db, hoi_classes_list, object_classes, verb_classes):
    all_anno_list = []

    for image_id, mat_anno in enumerate(mat_anno_db[0, :]):
        # print (image_id)

        # boxes: x1, y1, x2, y2
        image_anno = {'hboxes': [],
                      'oboxes': [],
                      'iboxes': [],
                      'hoi_classes': [],
                      'obj_classes': [],
                      'width': mat_anno['size']['width'][0, 0][0, 0],
                      'height': mat_anno['size']['height'][0, 0][0, 0],
                      'imageid': [],
                      'flipped': False}
        all_anno_list.append(image_anno)

        image_name = int(mat_anno['filename'][0].split('.')[0].split('_')[-1])
        image_anno['imageid'].append(image_name)
        # assert image_name == self._image_index[image_id]
        mat_hois = mat_anno['hoi'][0]
        # print len(mat_hois)
        for mat_hoi in mat_hois:
            visible = mat_hoi['invis'][0, 0] == 0
            if not visible:
                continue

            object_class2ind = dict(zip(object_classes, xrange(len(object_classes))))

            hoi_class_id = mat_hoi['id'][0, 0] - 1
            hoi_class = hoi_classes_list[hoi_class_id]
            object_name = hoi_class.object_name()
            # obj_class_id = object_class2ind[hoi_class.object_name()]
            obj_class_id = object_class2ind[object_name]

            human_boxes = mat_hoi['bboxhuman'][0]
            object_boxes = mat_hoi['bboxobject'][0]
            connections = mat_hoi['connection']
            for hid_oid in connections:
                hid, oid = hid_oid
                mat_hbox = human_boxes[hid - 1]
                hbox = [mat_hbox['x1'][0, 0] - 1, mat_hbox['y1'][0, 0] - 1,
                        mat_hbox['x2'][0, 0] - 1, mat_hbox['y2'][0, 0] - 1]
                mat_obox = object_boxes[oid - 1]
                obox = [mat_obox['x1'][0, 0] - 1, mat_obox['y1'][0, 0] - 1,
                        mat_obox['x2'][0, 0] - 1, mat_obox['y2'][0, 0] - 1]
                ibox = [min(hbox[0], obox[0]), min(hbox[1], obox[1]),
                        max(hbox[2], obox[2]), max(hbox[3], obox[3])]
                image_anno['hboxes'].append(hbox)
                image_anno['oboxes'].append(obox)
                image_anno['iboxes'].append(ibox)
                image_anno['hoi_classes'].append(hoi_class_id)
                image_anno['obj_classes'].append(obj_class_id)

        # list -> np.array
        if len(image_anno['hboxes']) == 0:
            image_anno['hboxes'] = np.zeros((0, 4))
            image_anno['oboxes'] = np.zeros((0, 4))
            image_anno['iboxes'] = np.zeros((0, 4))
            # image_anno['hoi_classes'] = np.zeros((0, len(hoi_classes)))
        else:
            hoi_classes = image_anno['hoi_classes']
            image_anno['hboxes'] = np.array(image_anno['hboxes'])
            image_anno['oboxes'] = np.array(image_anno['oboxes'])
            image_anno['iboxes'] = np.array(image_anno['iboxes'])
            # image_anno['hoi_classes'] = np.zeros((len(hoi_classes), len(hoi_classes)))
            # for i, cls in enumerate(hoi_classes):
            #     image_anno['hoi_classes'][i, cls] = 1

        # print ("huxw")

    return all_anno_list



sys.setrecursionlimit(100000)
anno = sio.loadmat('anno_bbox.mat')
mat_anno_db = anno['bbox_' + 'test']
mat_hoi_classes = anno['list_action']
hoi_cls_list, object_cls_list, verb_cls_list = load_hoi_classes(mat_hoi_classes)
all_anno_list = load_all_annotations(mat_anno_db, hoi_cls_list, object_cls_list, verb_cls_list)

dets_path = 'all_hoi_detections.pkl'
print('Loading  ...')
with open(dets_path) as f1:
    detts = pickle.load(f1)
print('Loading  over')
imid_det = detts.keys()
image_root = '/home/magus/dataset3/hico_20160224_det/images/test2015'
image_template = 'HICO_test2015_%s.jpg'

for i in range(len(imid_det)):
    key = imid_det[i]  # key is imid
    det = detts[key]

    len_ann = 0
    im_path = os.path.join(image_root,
                           image_template % str(key).zfill(8))

    # all_anno_list for imageid
    for t in range(len(all_anno_list)):
        ann = all_anno_list[t]
        if key != ann['imageid'][0]:
            continue

        hboxes_ann = ann['hboxes']
        oboxes_ann = ann['oboxes']
        hoi_classes_ann = ann['hoi_classes']
        obj_classes_ann = ann['obj_classes']
        len_ann = len(hboxes_ann)

        break

    # this is num for every image that detections
    for num in range(len(det)):
        hbox_det = det[num][0]
        obox_det = det[num][1]
        scores_det = det[num][3]
        hico = copy.deepcopy(scores_det)
        hico.sort(reverse=True)

        final_hoi_id = []
        final_objcls = []
        final_hbox_ann = []
        final_obox_ann = []
        s = 0
        verb = []
        obj = []

        for k in range(len_ann):
            hbox_ann = hboxes_ann[k]
            obox_ann = oboxes_ann[k]
            hoi_cls = hoi_classes_ann[k]
            obj_cls = obj_classes_ann[k]

            hbox_iou = iou(hbox_det, hbox_ann)
            obox_iou = iou(obox_det, obox_ann)

            if hbox_iou > 0.5 and obox_iou > 0.5:
                final_hbox_ann.append(hbox_ann)
                final_obox_ann.append((obox_ann))
                final_hoi_id.append(hoi_cls)
                final_objcls.append((obj_cls))
                s += 1

        for m in range(s):
            # score
            if scores_det[final_hoi_id[m]] in hico[:s]:
                hoi_class = hoi_cls_list[final_hoi_id[m]]

                verb_name =str_opera(hoi_class.verb_name())
                obj_name = str_opera(hoi_class.object_name())

                verb.append(verb_name)
                obj.append(obj_name)

        verbb = list(set(verb))
        objss = list(set(obj))

        vrbs = ', '.join(x for x in verbb)
        objs = ', '.join(x for x in objss)

        if len(verbb) > 1:
            show_boxes(im_path, str(key) + vrbs + str(num), [hbox_det, obox_det], [vrbs, objs], ['red', '#00B0F0'])
