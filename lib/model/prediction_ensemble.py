import datetime
import traceback

import numpy as np
import os
from configparser import ConfigParser

from lib.common.common_util import logging
from lib.model.generator import FeatruesSequence
from lib.model.build_model import ModelFactory
from keras.utils import multi_gpu_model
import tensorflow as tf


def set_sess_cfg():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


def main(openface_engineer_output_fp, openpose_engineer_output_fp, no_openpose_feature):
    video_name = os.path.join(os.path.basename(os.path.dirname(openface_engineer_output_fp)),
                              os.path.basename(openface_engineer_output_fp))
    print(f"start predict attetion:{video_name}")

    try:

        set_sess_cfg()
        # parser config
        config_file = "config.ini"
        cp = ConfigParser()
        cp.read(config_file)
        batch_size = cp["DEFAULT"].getint("batch_size")
        # parse weights file path
        output_weights_name = cp["DEFAULT"].get("output_weights_name")
        time_steps = cp["DEFAULT"].getint("time_steps")
        bidirect = cp["DEFAULT"].getboolean("bidirect")
        input_dim_openface = cp["DEFAULT"].getint("input_dim_openface")
        input_dim_openpose = cp["DEFAULT"].getint("input_dim_openpose")

        model_dirs_openface_lstm = [
            'bs16_seed2040_unitis16_layer3',
        ]
        model_dirs_openface_gru = [
            'bs16_seed2040_unitis128_layer1'
        ]
        model_dirs_openpose_lstm = [
            'bs16_seed2040_unitis32_layer2'
        ]
        model_dirs_openpose_gru = [
            'bs16_seed2040_unitis512_layer1'
        ]

        prob_array_list = []
        for featrue_name in ['openface', 'openpose']:
            if no_openpose_feature and featrue_name == 'openpose':
                continue
            for modle_mode in ['gru', 'lstm']:
                print("** load model **")
                model_dirs = eval(f'model_dirs_{featrue_name}_{modle_mode}')
                for model_dir in model_dirs:
                    weights_path = os.path.join('pretrained_models', featrue_name, modle_mode, model_dir,
                                                output_weights_name)
                    params = model_dir.split('_')
                    str_layers = params[-1]
                    str_units = params[-2]
                    num_layers = int(str_layers.split('layer')[1])
                    num_units = int(str_units.split('unitis')[1])
                    units_layers = []
                    [units_layers.append(num_units) for i in range(num_layers)]
                    # model_factory = ModelFactory()
                    model_factory = ModelFactory(predict=True)
                    model_fun = getattr(model_factory, f'get_model_{modle_mode}')
                    model = model_fun(
                        TIME_STEPS=time_steps,
                        INPUT_DIM=eval(f'input_dim_{featrue_name}'),
                        weights_path=weights_path,
                        CuDNN=False,
                        bidirect=bidirect,
                        units=units_layers,
                    )
                    print("** load test generator **")
                    test_sequence = FeatruesSequence(
                        features_name=featrue_name,
                        features_subdir=video_name,
                        batch_size=batch_size
                    )

                    gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(","))
                    if gpus > 1:
                        print(f"** multi_gpu_model is used! gpus={gpus} **")
                        model_predict = multi_gpu_model(model, gpus)
                    else:
                        model_predict = model

                    print("** make prediction **")
                    prob_array = model_predict.predict_generator(test_sequence,
                                                                 max_queue_size=8,
                                                                 workers=4,
                                                                 use_multiprocessing=False,
                                                                 verbose=1
                                                                 )
                    prob_array = np.squeeze(np.clip(prob_array, 0, 1))
                    prob_array_list.append(prob_array)
                    # df_results = pd.DataFrame()
                    # df_results['files'] = sorted(os.listdir(os.path.join('data/features/engineered', featrue_name, video_name)))
                    # df_results['probs'] = prob_array
                    # save_dir = os.path.join('results', video_name)
                    # if not os.path.exists(save_dir):
                    #     os.makedirs(save_dir, exist_ok=True)
                    # df_results.to_csv(os.path.join(save_dir, f'{featrue_name}_{modle_mode}.csv'), index=False)

        # prob_array_mean = np.mean(np.array(prob_array_list), axis=0)
        prob_array_mean = np.mean(prob_array_list, axis=0)
        # df_results_mean = pd.DataFrame()
        # df_results_mean['files'] = df_results['files']
        # df_results_mean['probs'] = prob_array_mean
        # df_results_mean.to_csv(os.path.join('results', video_name, 'results_mean.csv'), index=False)
        fixed_prob_mean = prob_array_mean
        if 0.8 > prob_array_mean > 0.2:
            fixed_prob_mean = (prob_array_mean - 0.2) / 0.6

        if no_openpose_feature:
            prob_array_list.extend([0, 0])

        return (fixed_prob_mean, prob_array_mean, prob_array_list)
    except Exception as e:
        logging(
            f"[prediction_ensemble.py][main|:{video_name}|:{openface_engineer_output_fp}][{datetime.datetime.now().strftime('%Y-%m-%d_%I:%M:%S')}]:"
            f"{traceback.format_exc()}",
            f"logs/error.log")


if __name__ == "__main__":
    set_sess_cfg()
    score, score_list = main(
        "/opt/Emotiw2021-Engagement-Prediction-KDDI/data/features/engineered/openface/7002_011/2021-03-03_08:16_2021-03-03_08:21.csv",
        "", True)
    print(
        f"mean - {score} openface_gru - {score_list[0]} openface_lstm - {score_list[1]} openpose_gru - {score_list[2]} openpose_lstm - {score_list[3]}")
