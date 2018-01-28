#!/usr/bin/env python3

from lxml import etree
import glob
import os
import uuid
import numpy as np
from PIL import Image
from os import listdir, getcwd
from os.path import join

classes = ['header', 'heading', 'page-number', 'paragraph', 'image']

ns = uuid.UUID('9f24ae6e-ea7c-52d2-9e97-9c028e640540')

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

if not os.path.exists('out'):
    os.makedirs('out/labels')
    os.makedirs('out/images')

def mk_bbox(s):
    """
    Makes a bounding box (x1, x2, y0, y1) out of a list of x,y coordinate tuples
    """
    xmin = 9999999999
    xmax = 0
    ymin = 9999999999
    ymax = 0
    for t in s.split():
        x, y = (int(u) for u in t.split(','))
        if x > xmax:
            xmax = x
        elif x < xmin:
            xmin = x
        if y > ymax:
            ymax = y
        elif y < ymin:
            ymin = y
    return (xmin, xmax, ymin, ymax)

manifest = []
for f in glob.iglob('data/**/*.xml', recursive=True):
    print('processing {}'.format(f))
    with open(f, 'rb') as fp:
        fid = uuid.uuid5(ns, fp.name)
        tree = etree.parse(fp)
        el = tree.find('{*}Page')
        im = os.path.join(os.path.dirname(f), el.get('imageFilename'))
        w = int(el.get('imageWidth'))
        h = int(el.get('imageHeight'))
        imf = 'out/images/{}.jpg'.format(fid)
        Image.open(im).save(imf)

        lf = 'out/labels/{}.txt'.format(fid)
        with open(lf, 'w') as fo:
            idx = 0
            for reg in el.findall('{*}TextRegion'):
                if reg.get('type') not in classes:
                    cls = classes.index('paragraph')
                else:
                    cls = classes.index(reg.get('type'))
                for tl in reg.findall('{*}TextLine'):
                    coords = tl.find('{*}Coords').get('points')
                    fo.write('{} {} {} {} {}\n'.format(cls, *convert((w, h), mk_bbox(coords))))
                    idx += 1
            cls = classes.index('image')
            for reg in el.findall('{*}ImageRegion'):
                coords = reg.find('{*}Coords').get('points')
                fo.write('{} {} {} {} {}\n'.format(cls, *convert((w, h), mk_bbox(coords))))
                idx += 1
        print('{} bboxes'.format(idx))
        manifest.append(imf)

np.random.shuffle(manifest)
with open('train.txt', 'w') as fp:
    fp.write('\n'.join(manifest[:-10]))

with open('test.txt', 'w') as fp:
    fp.write('\n'.join(manifest[-10:]))

with open('seg.names', 'w') as fp:
    fp.write('\n'.join(classes))

wd = getcwd()

with open('seg.data', 'w') as fp:
    fp.write(('classes = {0}\n'
              'train = {1}/train.txt\n'
              'test = {1}/test.txt\n'
              'names = {1}/seg.names\n'
              'backup = backup').format(len(classes), wd))
