import os
from configparser import ConfigParser

# load config.ini
config_file = "config.ini"
config = ConfigParser()
config.read(config_file)

output_target = config['face_feature_extraction_videos'].get('output_dir')
exec_path = config['face_feature_extraction_videos'].get('exec_path')
split_step = config['face_feature_extraction_videos'].getint('split_step')
start_cut = config['face_feature_extraction_videos'].getint('start_cut')
end_cut = config['face_feature_extraction_videos'].getint('end_cut')


def face_feature_extraction_videos(src, dst):
    dst_dir = os.path.dirname(dst)

    command = f"{exec_path} "
    command += f"-f {src} "
    command += f"-out_dir {os.path.dirname(dst)} -2Dfp -3Dfp -pdmparams -pose -aus -gaze"
    # command += f" -verbose"

    os.system(command)
    os.system(f"cd {dst_dir} && rm ./*.txt")

