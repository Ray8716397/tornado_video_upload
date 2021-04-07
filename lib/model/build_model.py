from tensorflow.keras.layers import LSTM, GRU, BatchNormalization, TimeDistributed, GlobalAveragePooling1D, Bidirectional
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential


class ModelFactory:

    def __init__(self, regularizers=None, initializer='orthogonal', with_FC_dropout_layers=True, rnn_dropout=0.2, predict=False):
        super(ModelFactory, self).__init__()
        self.regularizers = regularizers
        self.initializer = initializer
        self.with_FC_dropout_layers = with_FC_dropout_layers
        self.rnn_dropout = rnn_dropout
        if predict:
            self.with_FC_dropout_layers = False
            self.rnn_dropout = 0

            #lstm_units=[64, 64]
    def get_model_lstm(self, TIME_STEPS=150, INPUT_DIM=117, weights_path=None, units=[64, 64], CuDNN=False,
                       bidirect=False):
        model = Sequential()
        if bidirect:
            for i in range(len(units)):
                if i == 0:
                    model.add(Bidirectional(LSTM(
                        units[0],
                        return_sequences=True,
                        recurrent_initializer=self.initializer,
                        # kernel_regularizer=self.regularizers,
                        # recurrent_regularizer=self.regularizers,
                        # activity_regularizer=self.regularizers,
                        dropout=self.rnn_dropout,
                        recurrent_dropout=self.rnn_dropout),
                        input_shape=(TIME_STEPS, INPUT_DIM)))
                else:
                    model.add(Bidirectional(LSTM(
                        units[i],
                        return_sequences=True,
                        recurrent_initializer=self.initializer,
                        # kernel_regularizer=self.regularizers,
                        # recurrent_regularizer=self.regularizers,
                        # activity_regularizer=self.regularizers,
                        dropout=self.rnn_dropout,
                        recurrent_dropout=self.rnn_dropout)))
        else:
            for i in range(len(units)):
                if i == 0:
                    model.add(LSTM(
                        units[0],
                        return_sequences=True,
                        recurrent_initializer=self.initializer,
                        # kernel_regularizer=self.regularizers,
                        # recurrent_regularizer=self.regularizers,
                        # activity_regularizer=self.regularizers,
                        input_shape=(TIME_STEPS, INPUT_DIM),
                        dropout=self.rnn_dropout,
                        recurrent_dropout=self.rnn_dropout))
                else:
                    model.add(LSTM(units[i],
                                   return_sequences=True,
                                   recurrent_initializer=self.initializer,
                                   # kernel_regularizer=self.regularizers,
                        # recurrent_regularizer=self.regularizers,
                        # activity_regularizer=self.regularizers,
                                   dropout=self.rnn_dropout,
                                   recurrent_dropout=self.rnn_dropout))
        model.add(TimeDistributed(Dense(1024, name="fc1", kernel_regularizer=self.regularizers)))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        # add 20190522
        if self.with_FC_dropout_layers:
            model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(Dense(512, name="fc2", kernel_regularizer=self.regularizers)))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        # add 20190522
        if self.with_FC_dropout_layers:
            model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(Dense(128, name="fc3", kernel_regularizer=self.regularizers)))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        # add 20190522
        if self.with_FC_dropout_layers:
            model.add(TimeDistributed(Dropout(0.5)))
        # model.add(TimeDistributed(Dense(1, activation="sigmoid", name="predict")))
        model.add(TimeDistributed(Dense(1, name="predict")))
        model.add(GlobalAveragePooling1D())
        # model.add(Dense(1, activation='sigmoid'))

        if weights_path is not None:
            print(f"load model weights_path: {weights_path}")
            model.load_weights(weights_path)
        return model

    #gru_units=512
    def get_model_gru(self, TIME_STEPS=150, INPUT_DIM=117, weights_path=None, units=[512], CuDNN=False,
                      bidirect=False):
        model = Sequential()
        if bidirect:
            for i in range(len(units)):
                if i == 0:
                    model.add(Bidirectional(GRU(
                        units[0],
                        return_sequences=True,
                        recurrent_initializer=self.initializer,
                        dropout=self.rnn_dropout,
                        recurrent_dropout=self.rnn_dropout,
                        reset_after=False
                    ),
                        input_shape=(TIME_STEPS, INPUT_DIM)))
                else:
                    model.add(Bidirectional(GRU(
                        units[i],
                        return_sequences=True,
                        recurrent_initializer=self.initializer,
                        dropout=self.rnn_dropout,
                        recurrent_dropout=self.rnn_dropout,
                        reset_after=False
                    )))
        else:
            for i in range(len(units)):
                if i == 0:
                    model.add(GRU(
                        units[0],
                        return_sequences=True,
                        recurrent_initializer=self.initializer,
                        input_shape=(TIME_STEPS, INPUT_DIM),
                        dropout=self.rnn_dropout,
                        recurrent_dropout=self.rnn_dropout,
                        reset_after=False
                    ))
                else:
                    model.add(GRU(
                        units[i],
                        return_sequences=True,
                        recurrent_initializer=self.initializer,
                        dropout=self.rnn_dropout,
                        recurrent_dropout=self.rnn_dropout,
                        reset_after=False
                    ))
        model.add(TimeDistributed(Dense(128, name="fc1", kernel_regularizer=self.regularizers)))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        if self.with_FC_dropout_layers:
            model.add(TimeDistributed(Dropout(0.5)))
        # model.add(TimeDistributed(Dense(1, activation="sigmoid", name="predict")))
        model.add(TimeDistributed(Dense(1, name="predict")))
        model.add(GlobalAveragePooling1D())
        if weights_path is not None:
            print(f"load model weights_path: {weights_path}")
            model.load_weights(weights_path)
        return model