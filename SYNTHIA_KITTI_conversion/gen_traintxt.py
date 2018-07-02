#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate  kitti/image_sets/train.txt 

Created on Mon Jun 18 17:04:22 2018

@author: sohom31
"""

fp = open("train.txt",'w')

for frame in range(0, 150):
    print("Reading Frame no:", frame, "\n")
    
    file_no = '%06d' % frame
    
    fp.write(file_no)
    
    fp.write('\n')
    
fp.close()
    