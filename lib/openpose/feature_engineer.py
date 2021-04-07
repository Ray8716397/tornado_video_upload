import pandas as pd
from configparser import ConfigParser


# load config.ini
config_file = "config.ini"
config = ConfigParser()
config.read(config_file)

processes_num = config["DEFAULT"].getint("processes_num")
window_width = config["DEFAULT"].getint("window_width")
scale_lmk_openpose_x = config["DEFAULT"].getfloat("scale_lmk_openpose_x")
scale_lmk_openpose_y = config["DEFAULT"].getfloat("scale_lmk_openpose_y")
cols_names_all = ['x-0', 'y-0', 'x-1', 'y-1', 'x-2', 'y-2', 'x-3', 'y-3', 'x-4', 'y-4', 'x-5', 'y-5', 'x-6', 'y-6',
                  'x-7', 'y-7', 'x-15', 'y-15', 'x-16', 'y-16', 'x-17', 'y-17', 'x-18', 'y-18']



def extract_featrues_by_fixed_step(df, num_segment=150, step=12):
    list_sampled = []
    step_all = len(df)
    [list_sampled.append(df.iloc[i]) for i in range(0, step_all, step)]
    # assert len(list_sampled) == num_segment
    df_sampled = pd.DataFrame(data=list_sampled, columns=df.columns)
    return df_sampled


def pose_feature_engineer(src, dst):
    # 读取一个OpenPose_featrue文件
    df_OpenPose = pd.read_csv(src)
    columns_name_x = [i for i in df_OpenPose.columns if i[0] == 'x']
    columns_name_y = [i for i in df_OpenPose.columns if i[0] == 'y']
    df_OpenPose_lmk_x = df_OpenPose.loc[:, columns_name_x] / scale_lmk_openpose_x
    df_OpenPose_lmk_y = df_OpenPose.loc[:, columns_name_y] / scale_lmk_openpose_y

    # 链接成大的DF,求std和变化范围
    df_OpenPose = pd.concat([df_OpenPose_lmk_x, df_OpenPose_lmk_y], ignore_index=True, axis=1)
    df_OpenPose_std_variation_rolling = df_OpenPose.rolling(window=window_width)
    df_OpenPose_rolling_std = df_OpenPose_std_variation_rolling.std()
    df_OpenPose_rolling_variation = df_OpenPose_std_variation_rolling.max() - df_OpenPose_std_variation_rolling.min()
    # 合并所有特征为总特征
    df_all_feats = pd.concat([df_OpenPose_rolling_std, df_OpenPose_rolling_variation], ignore_index=True, axis=1)
    # 把df_all_feats_cols_name赋值给df_all_feats的列名
    columns_name_openpose = df_OpenPose_lmk_x.columns.tolist() + df_OpenPose_lmk_y.columns.tolist()
    columns_name_openpose_std = [column_name + '_std' for column_name in columns_name_openpose]
    columns_name_openpose_variation = [column_name + '_variation' for column_name in columns_name_openpose]
    df_all_feats.columns = columns_name_openpose_std + columns_name_openpose_variation
    # 按step=12,把5分钟视频分割成150小段，抽取特征
    df_all_feats.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    df_result = extract_featrues_by_fixed_step(df_all_feats, step=window_width)
    df_result.to_csv(dst, index=False)
