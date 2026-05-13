import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
KITTI_ROOT = os.path.join(PROJECT_ROOT, "data", "kitti")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "monodepth2", "splits")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
