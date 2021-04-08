# coding=utf-8
# @CREATE_TIME: 2021/4/8 上午10:24
# @LAST_MODIFIED: 2021/4/8 上午10:24
# @FILE: probResult.py
# @AUTHOR: Ray
import datetime
import glob
import os
import numpy as np


def get_today_min_res(user_id):
    # get_today_min_res
    today_min_res = []
    today = datetime.datetime.now()
    for prob_log_fp in sorted(glob.glob(f"logs/{user_id}/{today.strftime('%Y-%m-%d')}*"), reverse=True):
        with open(prob_log_fp, 'r') as f:
            lines = f.readlines()
            mean = float(lines[1].replace('mean - ', '').replace('\n', ''))
            fn = os.path.basename(prob_log_fp)
            record_time = fn[:16] + ' ~ ' + fn[17:]
            today_min_res.append({"record_time": record_time,
                                  "score": f"{format(mean, '.4f')}"})

    return today_min_res


def get_all_hour_res(user_id):
    # get_all_hour_res
    all_hour_res = []
    means = []
    prob_logs = sorted(glob.glob(f"logs/{user_id}/*"), reverse=True)
    if len(prob_logs) > 0:
        cur_hour_date = os.path.basename(prob_logs[0])[:13]
        for prob_log_fp in prob_logs:
            if os.path.isdir(prob_log_fp):
                continue
            with open(prob_log_fp, 'r') as f:
                lines = f.readlines()
                float_mean = float(lines[1].replace('mean - ', '').replace('\n', ''))
                means.append(float_mean)

            start_time = os.path.basename(prob_log_fp)[:13]
            if cur_hour_date != start_time and prob_log_fp != prob_logs[-1]:
                buffer_means = means[:-1]
                means = means[-1:]
            elif cur_hour_date == start_time and prob_log_fp == prob_logs[-1]:
                buffer_means = means
            elif cur_hour_date != start_time and prob_log_fp == prob_logs[-1]:
                buffer_means = means[-1:]
                temp_means = means[:-1]
                temp_means = [i for i in temp_means if i != 0.0]
                mean_result = np.array(temp_means).mean() if len(temp_means) > 0 else 0.0
                end_time = (datetime.datetime.strptime(cur_hour_date, "%Y-%m-%d_%H") + datetime.timedelta(
                    hours=1)).strftime('%Y-%m-%d_%H:%M')
                all_hour_res.append({"record_time": f"{cur_hour_date}:00 ~ {end_time}",
                                     "score": f"{format(mean_result, '.4f')}"})
                cur_hour_date = start_time
            else:
                continue

            buffer_means = [i for i in buffer_means if i != 0.0]
            mean_result = np.array(buffer_means).mean() if len(buffer_means) > 0 else 0.0
            end_time = (datetime.datetime.strptime(cur_hour_date, "%Y-%m-%d_%H") + datetime.timedelta(
                hours=1)).strftime('%Y-%m-%d_%H:%M')
            all_hour_res.append({"record_time": f"{cur_hour_date}:00 ~ {end_time}",
                                 "score": f"{format(mean_result, '.4f')}"})

            cur_hour_date = start_time

    return all_hour_res
