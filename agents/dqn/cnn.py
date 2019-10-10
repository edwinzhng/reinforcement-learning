import tensorflow as tf
from tensorflow.keras import layers


class ConvolutionalNetwork:
    def __init__(self, action_space_size: int, frame_skip: int):
        self.action_space_size = action_space_size
        self.frame_skip = frame_skip
        self.model = self.build_model()

    # model from Playing Atari with Deep Reinforcement Learning (Minh, 2015)
    def build_model(self):
        input = tf.keras.Input(shape=(84, 84, self.frame_skip), name='input')
        model = layers.Conv2D(filters=16, kernel_size=(8, 8),
                              strides=4, name='conv_1', activation='relu')(input)
        model = layers.Conv2D(filters=32, kernel_size=(4, 4),
                              strides=2, name='conv_2', activation='relu')(model)
        model = layers.Dense(256, activation='relu', name='fc_1')(model)
        output = layers.Dense(self.action_space_size, activation='softmax', name='output')(model)
        return tf.keras.Model(inputs=input, outputs=output)
