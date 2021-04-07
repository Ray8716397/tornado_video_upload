import glob
import json
import os
import shutil
from configparser import ConfigParser

import pandas as pd

# load config.ini
config_file = "config.ini"
config = ConfigParser()
config.read(config_file)

exec_path = config['pose_feature_extraction_videos'].get('exec_path')
openpose_dir = config['pose_feature_extraction_videos'].get('openpose_dir')
split_step = config['pose_feature_extraction_videos'].getint('split_step')
start_cut = config['pose_feature_extraction_videos'].getint('start_cut')
end_cut = config['pose_feature_extraction_videos'].getint('end_cut')


def json_featrue2csv(json_dir, output_csv_path):
    """
    openpose的json特征转csv
    :param json_dir: 存放特征的json文件夹路径
    :param output_csv_path: 生成的csv路径
    :return:
    """
    columns_name = []
    empty_row = [0 for i in range(24)]
    for i in range(8 + 4):
        if i < 8:
            columns_name.extend([f"x-{i}", f"y-{i}"])
        else:
            columns_name.extend([f"x-{i + 7}", f"y-{i + 7}"])

    if os.path.isdir(os.path.join(json_dir)):
        df = pd.DataFrame(data=[], columns=columns_name)

        for json_feature_path in sorted(glob.glob(json_dir + "/*.json")):
            with open(json_feature_path, 'rb') as json_file:
                json_dict = json.load(json_file)

            # check all the coordinate value of eyes and nose
            if len(json_dict["people"]) > 0:
                temp_array = []
                for people_num in range(len(json_dict["people"])):
                    x1, y1 = json_dict["people"][people_num]["pose_keypoints_2d"][:2]
                    x2, y2 = json_dict["people"][people_num]["pose_keypoints_2d"][15 * 3:16 * 3 - 1]
                    x3, y3 = json_dict["people"][people_num]["pose_keypoints_2d"][16 * 3:17 * 3 - 1]

                    if [x1, y1] != [0, 0] and [x2, y2] != [0, 0] and [x3, y3] != [0, 0]:
                        temp_array.append(json_dict["people"][people_num])

                json_dict["people"] = temp_array

            # choose the biggest face
            if len(json_dict["people"]) > 0:
                if len(json_dict["people"]) > 1:

                    area_array = []
                    for people_num in range(len(json_dict["people"])):
                        x1, y1 = json_dict["people"][people_num]["pose_keypoints_2d"][:2]
                        x2, y2 = json_dict["people"][people_num]["pose_keypoints_2d"][15 * 3:16 * 3 - 1]
                        x3, y3 = json_dict["people"][people_num]["pose_keypoints_2d"][16 * 3:17 * 3 - 1]

                        area = 0.5 * abs(x2 * y3 + x1 * y2 + x3 * y1 - x3 * y2 - x2 * y1 - x1 * y3)
                        area_array.append(area)

                    full_feature = json_dict["people"][area_array.index(max(area_array))]["pose_keypoints_2d"]

                else:
                    full_feature = json_dict["people"][0]["pose_keypoints_2d"]

                result_feature = full_feature[:8 * 3]
                result_feature.extend(full_feature[15 * 3:(18 + 1) * 3])
                result_feature = [i for index, i in enumerate(result_feature) if (index + 1) % 3 != 0]
                df.loc[df.shape[0]] = result_feature

            else:
                df.loc[df.shape[0]] = empty_row

        df.to_csv(output_csv_path, index=False)


def pose_feature_extraction_videos(src, dst):
    output_json_dirpath = dst.split('.')[0]
    os.system(f"cd {openpose_dir} && {exec_path} "
              f"--display 0 --render_pose 0 "
              f"--video {src} "
              f"--write_json {output_json_dirpath}")

    # json to csv
    json_featrue2csv(output_json_dirpath, output_json_dirpath + ".csv")
    # delete json features
    shutil.rmtree(output_json_dirpath)


if __name__ == '__main__':
    json_featrue2csv('/home/ray/Workspace/3rd_party_lib/openpose/out_temp2', "test.csv")
