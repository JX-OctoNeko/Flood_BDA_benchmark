#!/usr/bin/env bash

import sys
from tqdm import tqdm
from glob import glob
from itertools import count
from os import makedirs
from skimage.io import imread, imsave
from os.path import join, basename, splitext, exists
import shutil

CROP_SIZE = 256
STRIDE = 256
# CROP_SIZE = 128
# STRIDE = 128

if __name__ == '__main__':
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]

    for subset in ('train', 'test', 'hold'):
        for path in tqdm(glob(join(data_dir, subset, 'images', '*pre*.png'))):
            name, ext = splitext(basename(path))
            if subset == 'test':
                out_subdir = join(out_dir, 'val', 'A').replace('_target', '')
            elif subset == 'hold':
                out_subdir = join(out_dir, 'test', 'A').replace('_target', '')
            else:
                out_subdir = join(out_dir, subset, 'A').replace('_target', '')
            if not exists(out_subdir):
                makedirs(out_subdir)
            shutil.copyfile(path, join(out_subdir, name.replace('_pre', '').replace('_target', '')) + ext)

        for path in tqdm(glob(join(data_dir, subset, 'images', '*post*.png'))):
            name, ext = splitext(basename(path))
            if subset == 'test':
                out_subdir = join(out_dir, 'val', 'B').replace('_target', '')
            elif subset == 'hold':
                out_subdir = join(out_dir, 'test', 'B').replace('_target', '')
            else:
                out_subdir = join(out_dir, subset, 'B').replace('_target', '')
            if not exists(out_subdir):
                makedirs(out_subdir)
            shutil.copyfile(path, join(out_subdir, name.replace('_post', '').replace('_target', '')) + ext)

        for path in tqdm(glob(join(data_dir, subset, 'targets', '*pre*.png'))):
            name, ext = splitext(basename(path))
            im1 = (imread(path)).astype('uint8')
            im2 = (imread(path.replace('pre', 'post'))).astype('uint8')
            mask = im2 - im1
            if subset == 'test':
                out_subdir = join(out_dir, 'val', 'label').replace('_target', '')
            elif subset == 'hold':
                out_subdir = join(out_dir, 'test', 'label').replace('_target', '')
            else:
                out_subdir = join(out_dir, subset, 'label').replace('_target', '')
            if not exists(out_subdir):
                makedirs(out_subdir)
            imsave(join(out_subdir, name.replace('_pre', '').replace('_target', '')) + ext, mask)

