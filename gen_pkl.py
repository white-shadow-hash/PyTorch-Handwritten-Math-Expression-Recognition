'''
Python 3.6 
Pytorch 0.4
Written by Hongyu Wang in Beihang university
'''
import os
import sys
import pandas as pd
import pickle as pkl
import numpy
import imageio.v2 as iio
from pathlib import Path
outFile = 'offline-test.pkl'
oupFp_feature = open(outFile, 'wb')
features={}
channels = 1
sentNum = 0
caption_path = Path('./test_caption.txt')
if not caption_path.exists():
    raise FileNotFoundError("Missing './test_caption.txt'. Ensure it exists in the project root.")

# Peek first key to detect correct image directory structure
with open(caption_path, 'r') as _f:
    first_line = _f.readline().strip()
    if not first_line:
        raise RuntimeError('test_caption.txt is empty')
    first_key = first_line.split('\t')[0]

# Try common extraction layouts
candidates = [
    Path('./off_image_test/off_image_test/'),
    Path('./off_image_test/'),
]
image_path = None
for cand in candidates:
    probe = cand / f"{first_key}_0.bmp"
    if probe.exists():
        image_path = cand
        break

if image_path is None:
    raise FileNotFoundError(
        "Test images not found. Extract 'off_image_test/off_image_test.rar' so that .bmp files are present. "
        "Expected one of: './off_image_test/off_image_test/' or './off_image_test/'."
    )

scpFile = open(caption_path)
while 1:
    line=scpFile.readline().strip() # remove the '\r\n'
    if not line:
        break
    else:
        key = line.split('\t')[0]
        image_file = image_path / f"{key}_0.bmp"
        im = iio.imread(str(image_file))
        mat = numpy.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8')
        for channel in range(channels):
            image_file = image_path / f"{key}_{channel}.bmp"
            im = iio.imread(str(image_file))
            mat[channel,:,:] = im
        sentNum = sentNum + 1
        features[key] = mat
        if sentNum / 500 == sentNum * 1.0 / 500:
            print('process sentences ', sentNum)
print('load images done. sentence number ',sentNum)
pkl.dump(features,oupFp_feature)
print('save file done')
oupFp_feature.close()
