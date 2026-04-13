import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
from sklearn.pipeline import Pipeline
import pickle

import os
import sys

from mp_movement_classifier.utils import config
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.viz_tools import *


DEFAULT_DATA_DIR = "../../data/filtered_bvh_files"
SAVE_DIR = "../../data/expmap_files"


p = BVHParser()
BVH2expmap = MocapParameterizer('expmap')

bvh_files = [f for f in os.listdir(DEFAULT_DATA_DIR) if f.lower().endswith('.bvh')]

# bvh_filename = "subject_12_motion_06"
# bvh_file = f"../../data/filtered_bvh_files/{bvh_filename}.bvh"

for file in bvh_files:
    df = pd.DataFrame()
    data = [p.parse(os.path.join(DEFAULT_DATA_DIR, file))]
    xx = BVH2expmap.fit_transform(data)
    df = xx[0].values
    df.to_csv(f"{SAVE_DIR}/{file}.csv", index=True)