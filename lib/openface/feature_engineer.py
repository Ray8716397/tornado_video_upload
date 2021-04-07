import os
import pandas as pd
from configparser import ConfigParser

# load config.ini
config_file = "config.ini"
config = ConfigParser()
config.read(config_file)

processes_num = config["DEFAULT"].getint("processes_num")
window_width = config["DEFAULT"].getint("window_width")
scale_lmk = config["DEFAULT"].getint("scale_lmk")
scale_AU_r = config["DEFAULT"].getint("scale_AU_r")
scale_pose_T = config["DEFAULT"].getint("scale_pose_T")

# get_cols_names
columns_gaze = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x',
                'gaze_angle_y']
columns_pose = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']
columns_eye_head_lmk = ['eye_lmk_x', 'eye_lmk_y', 'eye_lmk_X', 'eye_lmk_Y', 'eye_lmk_Z', 'head_lmk_x', 'head_lmk_y',
                        'head_lmk_X', 'head_lmk_Y', 'head_lmk_Z']
columns_AU_r = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r',
                'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
columns_AU_c = ['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c',
                'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']
columns_std_variation = columns_gaze + columns_pose + columns_eye_head_lmk + columns_AU_r
df_std_cols_name = list(map(lambda column_name: column_name + '_std', columns_std_variation))
# df_std_cols_name = [column_name + '_std' for column_name in columns_std_variation]
df_variation_cols_name = list(map(lambda column_name: column_name + '_variation', columns_std_variation))
AU_r_rolling_max_columns = list(map(lambda column: column + '_max', columns_AU_r))
AU_c_rolling_frequency_columns = list(map(lambda column: column + '_frequency', columns_AU_c))
df_all_feats_cols_name = df_std_cols_name + df_variation_cols_name + AU_r_rolling_max_columns + AU_c_rolling_frequency_columns


def extract_featrues_by_fixed_step(df, num_segment=150, step=12):
    list_sampled = []
    step_all = len(df)
    [list_sampled.append(df.iloc[i]) for i in range(0, step_all, step)]
    df_sampled = pd.DataFrame(data=list_sampled[:num_segment], columns=df.columns)
    return df_sampled


def face_feature_engineer(src, dst):
    # 取得文件名
    openface_featrue_file_name = os.path.split(src)[-1]
    # 读取一个openface_featrue文件
    df_openface = pd.read_csv(src)
    df_openface.columns = df_openface.columns.str.strip()
    # 按gaze,eye_lmk_x,eye_lmk_y,eye_lmk_X,eye_lmk_Y,eye_lmk_Z,pose,head_lmk_x,head_lmk_y,head_lmk_X,head_lmk_Y,
    # head_lmk_Z,AU_r,AU_c分割成几个子DF,再进行进一步处理
    df_openface_gaze = df_openface.loc[:, 'gaze_0_x':'gaze_angle_y']
    df_openface_eye_lmk_x = df_openface.loc[:, 'eye_lmk_x_0':'eye_lmk_x_55'] / scale_lmk
    df_openface_eye_lmk_y = df_openface.loc[:, 'eye_lmk_y_0':'eye_lmk_y_55'] / scale_lmk
    df_openface_eye_lmk_X = df_openface.loc[:, 'eye_lmk_X_0':'eye_lmk_X_55'] / scale_lmk
    df_openface_eye_lmk_Y = df_openface.loc[:, 'eye_lmk_Y_0':'eye_lmk_Y_55'] / scale_lmk
    df_openface_eye_lmk_Z = df_openface.loc[:, 'eye_lmk_Z_0':'eye_lmk_Z_55'] / scale_lmk
    df_openface_pose = df_openface.loc[:, 'pose_Tx':'pose_Rz']
    df_openface_pose.loc[:, 'pose_Tx':'pose_Tz'] = df_openface_pose.loc[:, 'pose_Tx':'pose_Tz'] / scale_pose_T
    df_openface_head_lmk_x = df_openface.loc[:, 'x_0':'x_67'] / scale_lmk
    df_openface_head_lmk_y = df_openface.loc[:, 'y_0':'y_67'] / scale_lmk
    df_openface_head_lmk_X = df_openface.loc[:, 'X_0':'X_67'] / scale_lmk
    df_openface_head_lmk_Y = df_openface.loc[:, 'Y_0':'Y_67'] / scale_lmk
    df_openface_head_lmk_Z = df_openface.loc[:, 'Z_0':'Z_67'] / scale_lmk
    df_openface_AU_r = df_openface.loc[:, 'AU01_r':'AU45_r'] / scale_AU_r
    df_openface_AU_c = df_openface.loc[:, 'AU01_c':'AU45_c']
    # 求这10个landmark的平均值作为特征
    feats_lmks = [df_openface_eye_lmk_x, df_openface_eye_lmk_y, df_openface_eye_lmk_X, df_openface_eye_lmk_Y,
                  df_openface_eye_lmk_Z, df_openface_head_lmk_x, df_openface_head_lmk_y, df_openface_head_lmk_X,
                  df_openface_head_lmk_Y, df_openface_head_lmk_Z]
    columns_name_lmks_means = ['eye_lmk_x', 'eye_lmk_y', 'eye_lmk_X', 'eye_lmk_Y', 'eye_lmk_Z', 'head_lmk_x',
                               'head_lmk_y', 'head_lmk_X', 'head_lmk_Y', 'head_lmk_Z']
    df_lmks_means = pd.DataFrame()
    for index, feats_lmk in enumerate(feats_lmks):
        df_lmks_means[columns_name_lmks_means[index]] = feats_lmk.mean(axis=1)
    # 链接成大的DF,求std和变化范围
    openface_std_variation_list = [df_openface_gaze, df_openface_pose, df_lmks_means, df_openface_AU_r]
    df_openface_std_variation = pd.concat(openface_std_variation_list, ignore_index=True, axis=1)
    df_openface_std_variation_rolling = df_openface_std_variation.rolling(window=window_width)
    df_openface_rolling_std = df_openface_std_variation_rolling.std()
    df_openface_rolling_variation = df_openface_std_variation_rolling.max() - df_openface_std_variation_rolling.min()
    # 求AU_r在rolling中的最大值
    df_AU_r_rolling_max = df_openface_AU_r.rolling(window=window_width).max()
    # 求AU_c在rolling中的出现频率
    df_AU_c_rolling_frequency = df_openface_AU_c.rolling(window=window_width).sum() / window_width
    # 合并所有特征为总特征
    df_all_feats = pd.concat(
        [df_openface_rolling_std, df_openface_rolling_variation, df_AU_r_rolling_max, df_AU_c_rolling_frequency],
        ignore_index=True, axis=1)
    # 把df_all_feats_cols_name赋值给df_all_feats的列名
    df_all_feats.columns = df_all_feats_cols_name
    # 按step=12,把5分钟视频分割成150小段，抽取特征
    df_all_feats.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    df_result = extract_featrues_by_fixed_step(df_all_feats, step=window_width)
    df_result.to_csv(dst, index=False)
