#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Truncate point clouds to 3 decimal places

Created on Tue Jun 19 12:54:49 2018

@author: sohom31
"""
import numpy as np

PC_ROOT = "velodyne/"

for frame in range(0, 150):
    print("Frame no:", frame, "\n")
    pc_path = PC_ROOT + '%06d.bin' % frame
    points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
    points = np.around(points, decimals = 3)

    points.astype('float32').tofile(pc_path)