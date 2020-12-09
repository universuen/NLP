from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, GlobalMaxPooling1D, GlobalAveragePooling1D, Input, Bidirectional, \
    concatenate, SpatialDropout1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import numpy as np


class Models:
    def __init__(self, max_seq_len, vocabulary_size, embedding_size, embedding_matrix):
        inp = Input(shape=(max_seq_len,))

        x = Embedding(vocabulary_size + 1, embedding_size, weights=[embedding_matrix], trainable=True)(inp)
        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])
        x = Dense(10, activation="relu")(x)

        a_b = Dense(1, activation="sigmoid")(x)
        c = Dense(3, activation="softmax")(x)

        self.model_a = Model(inputs=inp, outputs=a_b)
        self.model_b = Model(inputs=inp, outputs=a_b)
        self.model_c = Model(inputs=inp, outputs=c)

    def compile(self, summary=False):
        """
        编译模型
        :param summary:是否显示模型概要
        :return:
        """
        self.model_a.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        self.model_b.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model_c.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        if summary is False:
            self.model_a.summary()
            # self.model_b.summary()
            # self.model_c.summary()

    def train(self, model_name, training_x, training_labels, validation_x, validation_labels, epochs=10):
        """
        训练指定模型
        :param model_name: 模型名称，只能是'a', 'b', 'c'中的一个
        :param training_x: 训练用输入
        :param training_labels: 训练用输出
        :param validation_x: 验证用输入
        :param validation_labels: 验证用输出
        :param epochs: 迭代轮数
        :return:
        """
        # 选择模型
        assert model_name in ['a', 'b', 'c']
        if model_name == 'a':
            model = self.model_a
        elif model_name == 'b':
            model = self.model_b
        else:
            model = self.model_c
            training_labels = to_categorical(training_labels, num_classes=3)
            validation_labels = to_categorical(validation_labels, num_classes=3)

        # 训练
        model.fit(training_x, training_labels, validation_data=(validation_x, validation_labels), epochs=epochs)

    def _predict(self, model_name, x):
        assert model_name in ['a', 'b', 'c']
        if model_name in ['a', 'b']:
            if model_name == 'a':
                model = self.model_a
            else:
                model = self.model_b
            predictions = model.predict(x)
            predictions = predictions > 0.5
        else:
            model = self.model_c
            predictions = model.predict(x)
            predictions = np.argmax(predictions, axis=1)
        return predictions

    def evaluate(self, model_name, x, labels):
        # 选择模型
        assert model_name in ['a', 'b', 'c']
        if model_name in ['a', 'b']:
            if model_name == 'a':
                target_name = ['OFF', 'NOT']
            else:
                target_name = ['UNT', 'TIN']
        else:
            target_name = ['IND', 'OTH', 'GRP']
        # 预测
        predictions = self._predict(model_name, x)
        print(f'\t\t-------EVALUATION OF MODEL {model_name.upper()}-------')
        print(classification_report(labels, predictions, target_names=target_name))
