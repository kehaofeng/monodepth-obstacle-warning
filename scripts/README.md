# Data Cleaning Scripts

Only the data cleaning and dataset preparation scripts are intended for the
minimal course submission.

- `download_kitti.py`: download the selected KITTI raw data.
- `check_kitti.py`: check local KITTI file availability.
- `clean_kitti_subset.py`: remove samples with missing or low-quality LiDAR.
- `analyze_kitti_cleaning.py`: summarize cleaning results.
- `split_dataset.py`: create train/val/test splits.
- `make_file_list.py`: export split file lists.
- `plot_kitti_analysis.py`: generate basic dataset analysis plots.

Training, checkpoint comparison, depth-map export, and TensorBoard export
helpers are kept locally but ignored by git for this submission scope.
