import json
import numpy as NP
from matplotlib import pyplot as PLT
import pandas as pd

d = {}

with open('memory_log.json') as f:
    d = json.load(f)

# Cenários:

# self.memLogger.define_log("pcd")
# self.memLogger.define_log("pcd_colorless")

# self.memLogger.define_log("octree")
# self.memLogger.define_log("octree_colorless")
# self.memLogger.define_log("octree_open3d")
# self.memLogger.define_log("octree_open3d_colorless")

# self.memLogger.define_log("voxel_grid")
# self.memLogger.define_log("voxel_grid_colorless")
# self.memLogger.define_log("voxel_grid_open3d")
# self.memLogger.define_log("voxel_grid_open3d_colorless")

# self.memLogger.define_log("only_octree")
# self.memLogger.define_log("only_octree_colorless")
# self.memLogger.define_log("only_octree_open3d")
# self.memLogger.define_log("only_octree_open3d_colorless")

# self.memLogger.define_log("only_voxel_grid")
# self.memLogger.define_log("only_voxel_grid_colorless")
# self.memLogger.define_log("only_voxel_grid_open3d")
# self.memLogger.define_log("only_voxel_grid_open3d_colorless")

# self.memLogger.define_log("low_level_world")
# self.memLogger.define_log("low_level_world_colorless")
# self.memLogger.define_log("high_level_world")
# self.memLogger.define_log("high_level_world_colorless")

print(d['pcd'])


# converting json dataset from dictionary to dataframe
train = pd.DataFrame.from_dict(d['pcd'], orient='index')
train.reset_index(level=0, inplace=True)
print(train)
ax = train.plot.area()