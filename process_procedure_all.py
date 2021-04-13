# coding=utf-8
"""
@File    :   Preprocess_all.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
19-5-27 下午4:18   Ray        1.0         None
"""

# import lib
import datetime
import glob
import os
import traceback
import multiprocessing
import pandas as pd
from configparser import ConfigParser

from lib.common import common_util
from lib.common.common_util import logging
from lib.ffmpeg.util import cv_set_video_fps_res
from lib.openface.feature_engineer import face_feature_engineer
from lib.openface.feature_extraction import face_feature_extraction_videos
from lib.openpose.feature_engineer import pose_feature_engineer
from lib.model import prediction_ensemble
from lib.common.feature_process import split_csv

# load config.ini
from lib.openpose.feature_extraction import pose_feature_extraction_videos

config_file = "./config.ini"
config = ConfigParser()
config.read(config_file)

split_step = config['pose_feature_extraction_videos'].getint('split_step')
start_cut = config['pose_feature_extraction_videos'].getint('start_cut')
end_cut = config['pose_feature_extraction_videos'].getint('end_cut')
face_min_input_size = config['face_feature_engineer'].getint('min_input_size')
pose_min_input_size = config['pose_feature_engineer'].getint('min_input_size')
video_6fps_dirpath = config['video_format'].get('output_dir')
face_feature_dirpath = config['face_feature_extraction_videos'].get('output_dir')
pose_feature_dirpath = config['pose_feature_extraction_videos'].get('output_dir')
face_engineer_dirpath = config['face_feature_engineer'].get('output_dir')
pose_engineer_dirpath = config['pose_feature_engineer'].get('output_dir')

min_input_size_on = config['DEFAULT'].getboolean('min_input_size_on')


# def prediction():
#     input_target = config['DEFAULT'].get('OpenPose_features_engineer_fold')
#     output_target = config['DEFAULT'].get('OpenPose_features_engineer_fold')
#
#     for files in os.listdir(input_target):
#         if os.path.isdir(os.path.join(input_target, files)):
#             if not os.path.exists(os.path.join(output_target, files)):
#                 os.makedirs(os.path.join(output_target, files))
#             prediction_ensemble.main(files)
def open_pose_task(output_6pfs_fp, user_id, duration, start_time):
    # openpose feature extraction
    common_util.mkdir(os.path.join(pose_feature_dirpath, user_id))
    openpose_6fps_output_fp = os.path.join(pose_feature_dirpath, user_id,
                                           duration + '.csv')
    try:
        pose_feature_extraction_videos(output_6pfs_fp, openpose_6fps_output_fp)
    except Exception as e:
        logging(
            f"[process_procedure_all.py][pose_feature_extraction_videos|id:{user_id}|stime:{start_time}][{datetime.datetime.now().strftime('%Y-%m-%d_%I:%M:%S')}]:"
            f"{traceback.format_exc()}",
            f"logs/error.log")


def predict_attention(user_id, start_time, end_time):
    no_openpose_flag = None
    chunks = sorted(glob.glob(f"{config['video_format'].get('input_dir')}/*{start_time}*.{user_id}"))
    duration = f"{start_time}_{end_time}"

    # convert to 6fps
    try:
        common_util.mkdir(os.path.join(video_6fps_dirpath, user_id))
        output_6pfs_fp = os.path.join(video_6fps_dirpath, user_id, f"{duration}.mp4")
        cv_set_video_fps_res(chunks, output_6pfs_fp)
    except Exception as e:
        logging(
            f"[process_procedure_all.py][cv_set_video_fps_res|id:{user_id}|stime:{start_time}][{datetime.datetime.now().strftime('%Y-%m-%d_%I:%M:%S')}]:"
            f"{traceback.format_exc()}",
            f"logs/error.log")

    for chunk in chunks:
        os.remove(chunk)

    # openpose task
    openpose_6fps_output_fp = os.path.join(pose_feature_dirpath, user_id,
                                           duration + '.csv')
    openpose_p = multiprocessing.Process(target=open_pose_task, args=(output_6pfs_fp, user_id, duration, start_time))
    openpose_p.start()

    # openface feature extraction
    common_util.mkdir(os.path.join(face_feature_dirpath, user_id))
    openface_6fps_output_fp = os.path.join(face_feature_dirpath, user_id,
                                           duration + '.csv')
    try:
        face_feature_extraction_videos(output_6pfs_fp, openface_6fps_output_fp)
    except Exception as e:
        logging(
            f"[process_procedure_all.py][face_feature_extraction_videos|id:{user_id}|stime:{start_time}][{datetime.datetime.now().strftime('%Y-%m-%d_%I:%M:%S')}]:"
            f"{traceback.format_exc()}",
            f"logs/error.log")

    if min_input_size_on:
        df_6fps_face_features = pd.read_csv(openface_6fps_output_fp)
        if not os.path.exists(openface_6fps_output_fp) or \
                df_6fps_face_features[df_6fps_face_features[' success'] == 1].shape[0] < face_min_input_size:
            return 0, (0, 0, 0, 0)

    split_csv(openface_6fps_output_fp, split_step, start_cut, end_cut)

    openpose_p.join()

    if min_input_size_on:
        no_openpose_flag = not os.path.exists(openpose_6fps_output_fp) or pd.read_csv(openpose_6fps_output_fp).shape[
            0] < pose_min_input_size
    else:
        no_openpose_flag = not os.path.exists(openpose_6fps_output_fp) or pd.read_csv(openpose_6fps_output_fp).shape[
            0] == 0

    if no_openpose_flag is not None and not no_openpose_flag:
        split_csv(openpose_6fps_output_fp, split_step, start_cut, end_cut)

    # openface feature engineer
    common_util.mkdir(os.path.join(face_engineer_dirpath, user_id))
    openface_engineer_output_fp = os.path.join(face_engineer_dirpath, user_id,
                                               duration + '.csv')
    face_feature_engineer(openface_6fps_output_fp, openface_engineer_output_fp)

    # openpose feature engineer
    common_util.mkdir(os.path.join(pose_engineer_dirpath, user_id))
    openpose_engineer_output_fp = os.path.join(pose_engineer_dirpath, user_id,
                                               duration + '.csv')
    if not no_openpose_flag:
        pose_feature_engineer(openpose_6fps_output_fp, openpose_engineer_output_fp)

    # predict attention
    return prediction_ensemble.main(openface_engineer_output_fp, openpose_engineer_output_fp, no_openpose_flag)


if __name__ == "__main__":
    common_util.mkdir("/home/ray/Workspace/test/py_test/test")
    output_6pfs_fp = "/home/ray/Workspace/test/py_test/test.webm"
    openpose_6fps_output_fp = "/home/ray/Workspace/test/py_test/1/test.csv"
    pose_feature_extraction_videos(output_6pfs_fp, openpose_6fps_output_fp)
