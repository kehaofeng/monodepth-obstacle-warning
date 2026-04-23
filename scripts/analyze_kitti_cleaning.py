import pandas as pd

csv_path = r"E:\monodepth_project\data\kitti\kitti_subset_cleaned.csv"
df = pd.read_csv(csv_path)

print("总样本数:", len(df))
print("\n各序列样本数：")
print(df["sequence"].value_counts())

print("\n图像读取失败数量:", (~df["image_ok"]).sum())
print("点云读取失败数量:", (~df["lidar_ok"]).sum())

print("\n图像尺寸统计：")
print(df[["image_height", "image_width"]].drop_duplicates())

print("\n点云点数统计：")
print(df["lidar_points"].describe())

# 设定一个简单异常阈值：点云点数小于 100000 视为异常样本
abnormal_df = df[df["lidar_points"] < 100000]

print("\n异常样本数量（点云点数 < 100000）：", len(abnormal_df))

abnormal_path = r"E:\monodepth_project\data\kitti\kitti_subset_abnormal.csv"
abnormal_df.to_csv(abnormal_path, index=False, encoding="utf-8-sig")
print("异常样本已保存到：", abnormal_path)

# 删除异常样本，生成最终训练用 cleaned 数据
clean_df = df[
    (df["image_ok"] == True) &
    (df["lidar_ok"] == True) &
    (df["image_height"] > 0) &
    (df["image_width"] > 0) &
    (df["lidar_points"] >= 100000)
].copy()

final_clean_path = r"E:\monodepth_project\data\kitti\kitti_subset_final_cleaned.csv"
clean_df.to_csv(final_clean_path, index=False, encoding="utf-8-sig")

print("\n最终清洗后样本数：", len(clean_df))
print("最终清洗结果已保存到：", final_clean_path)