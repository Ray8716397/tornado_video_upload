import glob
import os
from configparser import ConfigParser

# load config.ini
import cv2

config_file = "config.ini"
config = ConfigParser()
config.read(config_file)

fps = config['video_format'].getint('fps')
resolution = config['video_format'].get('resolution')
chunk_num = config['h5record_video'].getint('chunk_num')
interval = config['h5record_video'].getint('interval')

target_frame_total = int(fps * chunk_num * (interval/1000))


def set_video_fps_res(src, dst):
    """
    转换视频的fps和分辨率
    :param src: 输入路径
    :param dst: 输出路径
    :return:
    """

    print(f"start set_video_fps_res({src})")
    os.system(f"ffmpeg -y -i {src} "
              f"-r {fps} "
              f"-s {resolution} "
              f"-strict -2 {dst}")
    print(f"set_video_fps_res finished({dst})")


def cv_set_video_fps_res(src, dst):
    """
    转换视频的fps和分辨率
    :param src: 输入路径,数组
    :param dst: 输出路径
    :param origin_fps: 原视频fps
    :return:
    """
    w = 0
    h = 0
    src_frame_total = 0
    for i in src:
        vid_cap = cv2.VideoCapture(i)
        if i == src[0]:
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while True:
            ret, frame = vid_cap.read()
            if ret is False:
                break
            src_frame_total += 1
        vid_cap.release()

    fourcc = 'mp4v'  # output video codec
    target_fps = fps
    target_frame_step = src_frame_total/target_frame_total

    vid_writer = cv2.VideoWriter(dst,
                                 cv2.VideoWriter_fourcc(*fourcc), target_fps, (w, h))

    target_frame_count = 0
    target_step_count = 0
    target_frame = 0

    for i in src:
        vid_cap = cv2.VideoCapture(i)
        while True:
            ret, frame = vid_cap.read()
            if ret is False:
                break
            else:
                if target_frame_count == target_frame:
                    vid_writer.write(frame)
                    target_step_count += 1
                    target_frame = round(target_frame_step * target_step_count)
                target_frame_count += 1
        vid_cap.release()


def webm2mp4(src, dst):
    """
        转换视频格式,webm->mp4
        :param src: 输入路径
        :param dst: 输出路径
        :return:
        """
    os.system(f"ffmpeg -i {src} -qscale 0 {dst}")


def frame_extraction_videos(src, dst):

    os.makedirs(dst)
    os.system(f"ffmpeg -y -i {src} "
              f"-q:v 2 {dst}-%d.jpg")

# cv_set_video_fps_res(glob.glob("/home/ray/Downloads/aaaa/*"), '/home/ray/Downloads/kk.mp4')