#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 17:02:34 2018

@author: sohom31
"""

import numpy as np

PC_ROOT = "velodyne/"
trash = 0

for frame in range(0, 8088):
    print("Frame no:", frame, "\n")
    pc_path = PC_ROOT + '%06d.bin' % frame
    points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
    
    if points.shape[0] == 0:
        trash = trash + 1
        
    print("Trash:", trash, "\n")
        
