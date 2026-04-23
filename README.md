# \# Monodepth Obstacle Warning

# 

# Course project on monocular depth estimation and obstacle warning using KITTI, Monodepth2, and Lite-Mono.

# 

# \## Project Structure

# 

# \- `data/kitti/`: KITTI subset metadata files

# \- `scripts/`: data cleaning, checking, and visualization scripts

# \- `results/plots/`: analysis figures

# \- `logs/`: training logs

# \- `weights/`: saved model checkpoints

# \- `notebooks/`: exploratory notebooks

# 

# \## Current Progress

# 

# \### Data Cleaning

# \- Selected 3 KITTI raw sequences as experimental subset:

# &nbsp; - `2011\_09\_26\_drive\_0001\_sync`

# &nbsp; - `2011\_09\_26\_drive\_0002\_sync`

# &nbsp; - `2011\_09\_26\_drive\_0005\_sync`

# \- Checked image-point cloud correspondence

# \- Verified file readability

# \- Detected abnormal samples based on LiDAR point counts

# \- Exported cleaned dataset files:

# &nbsp; - `kitti\_subset\_cleaned.csv`

# &nbsp; - `kitti\_subset\_cleaned.json`

# &nbsp; - `kitti\_subset\_final\_cleaned.csv`

# 

# \### Data Analysis

# Generated plots:

# \- `sequence\_counts\_bar.png`

# \- `lidar\_points\_hist.png`

# \- `lidar\_points\_line.png`

# \- `sample\_images\_grid.png`

# 

# \## Environment

# 

# Python 3.9 recommended.

# 

# \## Next Steps

# \- Build Monodepth2 baseline environment

# \- Run baseline training

# \- Compare with Lite-Mono

# \- Develop obstacle warning visualization

