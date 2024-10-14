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

if __name__ == '__main__':
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]

    for subset in ('train', 'test', 'hold'):
        for path in tqdm(glob(join(data_dir, subset, 'images', '*pre*.png'))):
            name, ext = splitext(basename(path))
            if subset == 'test':
                out_subdir = join(out_dir, 'val', 'A')
            elif subset == 'hold':
                out_subdir = join(out_dir, 'test', 'A')
            else:
                out_subdir = join(out_dir, subset, 'A')
            if not exists(out_subdir):
                makedirs(out_subdir)
            shutil.copyfile(path, join(out_subdir, name.replace('_pre', '')) + ext)

        for path in tqdm(glob(join(data_dir, subset, 'images', '*post*.png'))):
            name, ext = splitext(basename(path))
            if subset == 'test':
                out_subdir = join(out_dir, 'val', 'B')
            elif subset == 'hold':
                out_subdir = join(out_dir, 'test', 'B')
            else:
                out_subdir = join(out_dir, subset, 'B')
            if not exists(out_subdir):
                makedirs(out_subdir)
            shutil.copyfile(path, join(out_subdir, name.replace('_post', '')) + ext)

        for path in tqdm(glob(join(data_dir, subset, 'targets', '*pre*.png'))):
            name, ext = splitext(basename(path))
            im1 = (imread(path)*255).astype('int16')
            im2 = (imread(path.replace('pre', 'post'))*255).astype('int16')
            mask = im2 - im1
            # mask = im2
            # mask[mask==255] = 0
            # mask[mask>0] = 255
            mask[mask<0] = 0
            if subset == 'test':
                out_subdir = join(out_dir, 'val', 'label')
            elif subset == 'hold':
                out_subdir = join(out_dir, 'test', 'label')
            else:
                out_subdir = join(out_dir, subset, 'label')
            if not exists(out_subdir):
                makedirs(out_subdir)
            imsave(join(out_subdir, name.replace('_pre', '')) + ext, mask)

