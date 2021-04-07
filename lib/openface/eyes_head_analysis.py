from configparser import ConfigParser

import pandas as pd
import numpy as np
import argparse

# load config.ini
config_file = "config.ini"
config = ConfigParser()
config.read(config_file)

eye_open_rate_threshold = config["eyes_head_analysis"].getfloat("eye_open_rate_threshold")
gaze_angle_x_threshold = config["eyes_head_analysis"].getfloat("gaze_angle_x_threshold")
gaze_angle_y_threshold = config["eyes_head_analysis"].getfloat("gaze_angle_y_threshold")
pose_Rx_max = config["eyes_head_analysis"].getfloat("pose_Rx_max")
pose_Ry_max = config["eyes_head_analysis"].getfloat("pose_Ry_max")
pose_Rz_max = config["eyes_head_analysis"].getfloat("pose_Rz_max")
pose_Ry_min = config["eyes_head_analysis"].getfloat("pose_Ry_min")


# 检测眼镜是否睁开，当两眼的高/宽都大于0.22时，认为是睁眼返回True，否则认为闭眼返回False
def apply_eye_open(line):
    eyes_open = True
    left_eye_top = (line[0], line[1])
    left_eye_down = (line[2], line[3])
    left_eye_left = (line[4], line[5])
    left_eye_right = (line[6], line[7])
    right_eye_top = (line[8], line[9])
    right_eye_down = (line[10], line[11])
    right_eye_left = (line[12], line[13])
    right_eye_right = (line[14], line[15])
    left_eye_open_rate = (left_eye_down[1] - left_eye_top[1]) / (left_eye_right[0] - left_eye_left[0])
    right_eye_open_rate = (right_eye_down[1] - right_eye_top[1]) / (right_eye_right[0] - right_eye_left[0])
    if 0 < left_eye_open_rate < eye_open_rate_threshold and 0 < right_eye_open_rate < eye_open_rate_threshold:
        eyes_open = False
    return [eyes_open, left_eye_open_rate, right_eye_open_rate]


# 检测视角是否在一定范围内
def apply_gaze_in_range(line):
    gaze_in_range = True
    gaze_angle_x, gaze_angle_y = line[6], line[7]
    # 2021-03-08_15:01_2021-03-08_15:-0.7<gaze_angle_x<0.3,-0.357<gaze_angle_y<0.347
    # 2021-03-08_11:20_2021-03-08_11:25_closeeyes:-0.419<gaze_angle_x<0.098,-0.053<gaze_angle_y<0.441
    # 2021-03-05_19:22_2021-03-05_19:27:-0.9<gaze_angle_x<0.6,-0.475<gaze_angle_y<0.57
    # 2021-03-05_11:15_2021-03-05_11:20:-0.569<gaze_angle_x<0.369,-0.175<gaze_angle_y<0.498
    if abs(gaze_angle_x) > gaze_angle_x_threshold or abs(gaze_angle_y) > gaze_angle_y_threshold:
        gaze_in_range = False
    return gaze_in_range


# 检测面部角度是否在一定范围内
def apply_face_in_range(line):
    face_in_range = True
    pose_Rx, pose_Ry, pose_Rz = line[0], line[1], line[2]
    # 2021-03-08_15:01_2021-03-08_15:06_range:-0.573<x<0.379,-0.3<y<1.06,-0.292<z<0.388
    # 2021-03-08_11:20_2021-03-08_11:25_closeeyes:-0.308<x<0.298,-0.128<y<0.479,-0.346<z<0.11
    # 2021-03-05_19:22_2021-03-05_19:27:-0.701<x<0.472,-0.59<y<1.02,-0.25<z<0.4
    # 2021-03-05_11:15_2021-03-05_11:20:-0.42<x<0.528,-0.48<y<0.74,-0.5<z<0.179
    if abs(pose_Rx) > pose_Rx_max or pose_Ry > pose_Ry_max or pose_Ry < pose_Ry_min or abs(pose_Rz) > pose_Rz_max:
        face_in_range = False
    return face_in_range


def features_analysis(features_file):
    face_df = pd.read_csv(features_file)
    face_df.columns = face_df.columns.str.strip()
    left_eyes_top_down_left_right_columns_names = ['eye_lmk_x_11', 'eye_lmk_y_11', 'eye_lmk_x_17', 'eye_lmk_y_17',
                                                   'eye_lmk_x_8', 'eye_lmk_y_8', 'eye_lmk_x_14', 'eye_lmk_y_14']
    right_eyes_top_down_left_right_columns_names = ['eye_lmk_x_39', 'eye_lmk_y_39', 'eye_lmk_x_45', 'eye_lmk_y_45',
                                                    'eye_lmk_x_36', 'eye_lmk_y_36', 'eye_lmk_x_42', 'eye_lmk_y_42']
    gaze_columns_names = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x',
                          'gaze_angle_y']
    head_pose_columns_names = ['pose_Rx', 'pose_Ry', 'pose_Rz']
    # 检测到人脸的DF
    has_face_df = face_df[face_df['success'] == 1]
    count_total = face_df.shape[0]
    count_has_face = has_face_df.shape[0]
    # 有脸帧数占总帧数的比例
    rate_has_face = count_has_face / count_total
    print(f'count_total:{count_total},count_has_face:{count_has_face},rate_has_face:{rate_has_face}')
    # 检测是否闭眼开始------------------------------------------------------
    # 抽出左右两眼上下左右坐标的DF
    eyes_lmk_df = has_face_df[
        left_eyes_top_down_left_right_columns_names + right_eyes_top_down_left_right_columns_names]
    # 检测眼睛是否睁开
    eyes_lmk_df[['eyes_open', 'left_eye_open_rate', 'right_eye_open_rate']] = eyes_lmk_df.apply(apply_eye_open, axis=1,
                                                                                                result_type="expand")
    count_eyes_open = eyes_lmk_df[eyes_lmk_df['eyes_open'] == True].shape[0]
    # 睁眼帧数占有脸帧数的比例
    rate_eyes_open = count_eyes_open / count_has_face
    print(f'count_eyes_open:{count_eyes_open},rate_eyes_open:{rate_eyes_open}')
    # 检测是否闭眼结束------------------------------------------------------

    # 检测面部角度开始------------------------------------------------------
    # 抽出面部角度的DF
    head_pose_df = has_face_df[head_pose_columns_names]
    # 检测面部角度是否在一定范围内
    head_pose_df['face_in_range'] = head_pose_df.apply(apply_face_in_range, axis=1)
    count_face_in_range = head_pose_df[head_pose_df['face_in_range'] == True].shape[0]
    # 面部角度正常帧数占有脸帧数的比例
    rate_face_in_range = count_face_in_range / count_has_face
    print(f'count_face_in_range:{count_face_in_range},rate_face_in_range:{rate_face_in_range}')
    # 检测面部角度结束------------------------------------------------------

    # 检测视角开始------------------------------------------------------
    # 抽出视角的DF
    gaze_df = has_face_df[gaze_columns_names]
    # 检测视角是否在一定范围内
    gaze_df['gaze_in_range'] = gaze_df.apply(apply_gaze_in_range, axis=1)
    count_gaze_in_range = gaze_df[gaze_df['gaze_in_range'] == True].shape[0]
    # 视角正常帧数占有脸帧数的比例
    rate_gaze_in_range = count_gaze_in_range / count_has_face
    print(f'count_gaze_in_range:{count_gaze_in_range},rate_gaze_in_range:{rate_gaze_in_range}')
    # 检测视角结束------------------------------------------------------
    return rate_has_face, rate_eyes_open, rate_face_in_range, rate_gaze_in_range


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 2021-03-08_15:01_2021-03-08_15:06_range,2021-03-08_11:20_2021-03-08_11:25_closeeyes,2021-03-05_19:22_2021-03-05_19:27,2021-03-05_19:05_2021-03-05_19:10,2021-03-05_18:59_2021-03-05_19:04,2021-03-05_11:15_2021-03-05_11:20
    parser.add_argument('--features_file', type=str,
                        default='data/features/6fps/openface/7002_003/2021-03-05_11:15_2021-03-05_11:20.csv',
                        help='features file path to analysis')
    opt = parser.parse_args()
    rate_has_face, rate_eyes_open, rate_face_in_range, rate_gaze_in_range = features_analysis(opt.features_file)
