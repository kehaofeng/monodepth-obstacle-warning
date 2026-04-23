# Monodepth Obstacle Warning

Course project on monocular depth estimation and obstacle warning using KITTI, Monodepth2, and Lite-Mono.

## Project Structure

- `data/kitti/`: KITTI subset metadata files
- `scripts/`: data cleaning, checking, and visualization scripts
- `results/plots/`: analysis figures
- `logs/`: training logs
- `weights/`: saved model checkpoints
- `notebooks/`: exploratory notebooks

## Current Progress

### Data Cleaning
- Selected 3 KITTI raw sequences as experimental subset:
  - `2011_09_26_drive_0001_sync`
  - `2011_09_26_drive_0002_sync`
  - `2011_09_26_drive_0005_sync`
- Checked image-point cloud correspondence
- Verified file readability
- Detected abnormal samples based on LiDAR point counts
- Exported cleaned dataset files:
  - `kitti_subset_cleaned.csv`
  - `kitti_subset_cleaned.json`
  - `kitti_subset_final_cleaned.csv`

### Data Analysis
Generated plots:
- `sequence_counts_bar.png`
- `lidar_points_hist.png`
- `lidar_points_line.png`
- `sample_images_grid.png`

## Environment

Python 3.9 recommended.

## Next Steps
- Build Monodepth2 baseline environment
- Run baseline training
- Compare with Lite-Mono
- Develop obstacle warning visualization