import numpy as np
import os
from configparser import ConfigParser
from lib.model.generator import FeatruesSequence
from lib.model.build_model import ModelFactory
import pandas as pd
from keras.utils import multi_gpu_model
from utility import get_sample_counts, set_sess_cfg
from math import ceil
from keras import regularizers


def out_put_prediction(prob):
    if prob < 0.21:
        pred_list.append(0)
    elif prob <= 0.51:
        pred_list.append(0.33)
    elif prob <= 0.79:
        pred_list.append(0.66)
    else:
        pred_list.append(1)


def main():
    # parser config
    config_file = "../../config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    # test config
    output_dir = cp["TEST"].get("output_dir")
    batch_size = cp["TEST"].getint("batch_size")
    model_name = cp["TEST"].get("model_name")
    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir, output_weights_name)
    time_steps = cp["TRAIN"].getint("time_steps")
    input_dim_openface = cp["TRAIN"].getint("input_dim_openface")
    input_dim_openpose = cp["TRAIN"].getint("input_dim_openpose")
    input_dim_rgb400 = cp["TRAIN"].getint("input_dim_rgb400")
    csv_source_dir_openface = cp["DEFAULT"].get("csv_source_dir_openface")
    csv_source_dir_openpose = cp["DEFAULT"].get("csv_source_dir_openpose")
    csv_source_dir_rgb400 = cp["DEFAULT"].get("csv_source_dir_rgb400")
    featrue_name = cp["TEST"].get("featrue_name")
    cuDNN = cp["TRAIN"].getboolean("cuDNN")
    bidirect = cp["TEST"].getboolean("bidirect")
    units = cp["TEST"].getint("units")
    layers = cp["TEST"].getint("layers")
    with_FC_dropout_layers = cp["TEST"].getboolean("with_FC_dropout_layers")
    initializer = cp["TEST"].get("initializer")
    regularizers_fuc = cp["TEST"].get("regularizers_fuc")
    regularizers_l1 = cp["TEST"].getfloat("regularizers_l1")
    regularizers_l2 = cp["TEST"].getfloat("regularizers_l2")
    if regularizers_fuc == 'l1':
        regularizers_fuc = regularizers.l1(regularizers_l1)
    elif regularizers_fuc == 'l2':
        regularizers_fuc = regularizers.l2(regularizers_l2)
    elif regularizers_fuc == 'l1_l2':
        regularizers_fuc = regularizers.l1_l2(regularizers_l1, regularizers_l2)
    else:
        regularizers_fuc = None
    units_layers = []
    [units_layers.append(units) for i in range(layers)]
    test_csv = cp["TEST"].get("test_csv")

    # compute steps
    # test_counts = get_sample_counts("test")
    test_counts = get_sample_counts("val")
    test_steps = ceil(test_counts / batch_size)
    print(f"** test_steps: {test_steps} **")

    print("** load model **")
    model_factory = ModelFactory(regularizers=regularizers_fuc,
                                 initializer=initializer,
                                 with_FC_dropout_layers=with_FC_dropout_layers)
    model_fun = getattr(model_factory, f'get_model_{model_name}')
    model = model_fun(
        TIME_STEPS=time_steps,
        INPUT_DIM=eval(f'input_dim_{featrue_name}'),
        weights_path=weights_path,
        CuDNN=cuDNN,
        bidirect=bidirect,
        units=units_layers,
    )

    print("** load test generator **")
    test_sequence = FeatruesSequence(
        dataset_csv_file=os.path.join('data', test_csv),
        csv_source_dir=eval(f'csv_source_dir_{featrue_name}'),
        batch_size=batch_size,
        shuffle_on_epoch_end=False,
        test=True
    )

    gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(","))
    if gpus > 1:
        print(f"** multi_gpu_model is used! gpus={gpus} **")
        model_train = multi_gpu_model(model, gpus)
    else:
        model_train = model

    print("** make prediction **")
    # model_train.compile(optimizer=Adam(), loss="mean_squared_error")
    prob_array = model_train.predict_generator(test_sequence,
                                               steps=test_steps,
                                               max_queue_size=8,
                                               workers=8,
                                               use_multiprocessing=True,
                                               verbose=1
                                               )
    print(prob_array)
    list(map(out_put_prediction, prob_array))
    print(pred_list)
    df_test = pd.read_csv(os.path.join('data', test_csv))
    df_test['predictions'] = pred_list
    df_test['probs'] = prob_array
    df_test[['openface_file', 'predictions', 'probs']].to_csv(os.path.join(output_dir, 'result.csv'), index=False)


if __name__ == "__main__":
    np.set_printoptions(threshold=1e6)
    set_sess_cfg()
    pred_list = []
    main()
