import numpy as np
import sys
import os.path as osp
import random as rd
import multiprocessing
# import cv2
import os
import tqdm
import collections
import subprocess
from functools import reduce
from PIL import Image
import json
import pickle
import matplotlib.pyplot as plt
import hashlib
os.environ['PYTHONHASHSEED'] = '0'
from IPython.display import Video
import mmcv
import moviepy.editor as mpy

def mmap(func, *args):
    return list(map(func, *args))


def mfilter(func, *args):
    return list(filter(func, *args))


def mrlines(fname, sp='\n'):
    f = open(fname).read().split(sp)
    while f != [] and f[-1] == '':
        f = f[:-1]
    return f

def default_set(self, args, name, default):
    if hasattr(args, name):
        val = getattr(args, name)
        setattr(self, name, val)
    else:
        setattr(self, name, default)


def youtube_dl(url, output_name):
	cmd = 'youtube-dl -f best -f mp4 "{}"  -o {}'.format(url, output_name)
	os.system(cmd)

def jsonread(fname):
	return json.load(open(fname))

def npymd5(arr):
    hasher = hashlib.md5()
    hasher.update(str(arr.data).encode())
    return hasher.hexdigest()

def npysha1(arr):
    return hashlib.sha1(str(arr.data).encode()).hexdigest()

def run_command(cmd):
	return subprocess.check_output(cmd)


def ls(dirname='.', full=True):
    if not full:
        return os.listdir(dirname)
    return mmap(lambda x: osp.join(dirname, x), os.listdir(dirname))

def add(x, y):
    return x + y

def lpkl(pth):
    return pickle.load(open(pth, 'rb'))

def ljson(pth):
    return json.load(open(pth, 'r'))


def xstack_videos(vid_list, shape, out_path):
    # only support videos with same shape now
    assert len(shape) == 2
    assert len(vid_list) == shape[0] * shape[1]
    vid_list = mmap(lambda x: '-i ' + x, vid_list)
    cmd1 = 'ffmpeg'
    cmd2 = ' '.join(vid_list)
    cmd25 = '-vcodec h264 -filter_complex'
    cmd3 = []
    for i in range(shape[0]):
        inds = list(range(i * shape[1], (i + 1) * shape[1]))
        prefix = mmap(lambda x: '[{}:v]'.format(x), inds)
        prefix = ''.join(prefix)
        suffix = 'hstack=inputs={}[mid{}]'.format(shape[1], i)
        cmd3.append(prefix + suffix)
    prefix = mmap(lambda x: '[mid{}]'.format(x), range(shape[0]))
    prefix = ''.join(prefix)
    suffix = 'vstack=inputs={}[v]'.format(shape[0])
    cmd3.append(prefix + suffix)
    cmd3 = "\"" + ';'.join(cmd3) + "\""
    
    cmd4 = "-map \"[v]\" {}".format(out_path)
    cmd = ' '.join([cmd1, cmd2, cmd25, cmd3, cmd4])
    if osp.exists(out_path):
        os.remove(out_path)
    os.system(cmd)
    

def StackVideo(vid_list, shape):
    pth = '/tmp/output.mp4'
    xstack_videos(vid_list, shape, pth)
    vid = mpy.VideoFileClip(pth)
    return vid

def StackImage(image_list, shape, keep_scale=False):
    assert len(image_list) == shape[0] * shape[1]
    imgs = mmap(lambda x: mmcv.imread(x), image_list)
    if keep_scale:
        imgs = mmap(lambda x: mmcv.impad_center(mmcv.imrescale(x, (256, 256)), (256, 256), (128, 128, 128)), imgs)
    else:
        imgs = mmap(lambda x: mmcv.imresize(x, (256, 256)), imgs)
    horizon = mmap(lambda x: np.concatenate(imgs[x * shape[1]: (x + 1) * shape[1]], axis=1) ,range(shape[0]))
    full = np.concatenate(horizon, axis=0)
    img = mpy.ImageClip(full[:,:,::-1])
    return img


def intop(pred, label, n):
    pred = mmap(lambda x: np.argsort(x)[0][-n:], pred)
    hit = mmap(lambda l, p: l in p, label, pred)
    return hit
