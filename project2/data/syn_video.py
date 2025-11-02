# import cv2, os, pandas as pd

# DIR = r'D:\Master2_forward\CS290U\project2\data'
# df = pd.read_csv(os.path.join(DIR, 'rgb.txt'),
#                  sep=' ', comment='#', header=None, names=['t', 'file'])
# df.sort_values('t', inplace=True)

# for _, row in df.head(5).iterrows():
#     full = os.path.join(DIR, row['file'])
#     img = cv2.imread(full)
#     print(full, 'OK' if img is not None else 'FAILED')
import cv2, os, pandas as pd, tqdm

# ---------- 参数 ----------
DIR        = r'D:\Master2_forward\CS290U\project2\data'   # 数据集根目录
OUT_NAME   = 'TUM_freiburg1_desk2_secret.mp4'             # 输出视频
FPS        = 30                                           # 播放帧率
FOURCC     = cv2.VideoWriter_fourcc(*'mp4v')              # H264 编码
# 若需同步保存深度，取消下行注释
# SAVE_DEPTH = True
# --------------------------

# 读取 RGB 索引
rgb_df = pd.read_csv(os.path.join(DIR, 'rgb.txt'),
                     sep=' ', comment='#', header=None, names=['t', 'file'])
rgb_df.sort_values('t', inplace=True)

# 获得图像尺寸（用第一帧）
first = cv2.imread(os.path.join(DIR, rgb_df.iloc[0]['file']))
H, W = first.shape[:2]

# 创建视频写入器
vout = cv2.VideoWriter(os.path.join(DIR, OUT_NAME), FOURCC, FPS, (W, H))

print(f'Writing {len(rgb_df)} frames -> {OUT_NAME}')
for _, row in tqdm.tqdm(rgb_df.iterrows(), total=len(rgb_df)):
    img = cv2.imread(os.path.join(DIR, row['file']))
    vout.write(img)
    cv2.imshow('Preview (ESC to quit)', img)
    if cv2.waitKey(1) == 27:          # ESC 退出
        break

vout.release()
cv2.destroyAllWindows()
print('Done! 视频已保存到:', os.path.join(DIR, OUT_NAME))