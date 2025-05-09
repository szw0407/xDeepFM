import numpy as np
import pandas as pd
import tensorflow as tf
import Config
from tools import _get_data, _get_conf, get_label, auc_score
import time

class xDeepFM(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.total_emb, self.single_size, self.numerical_size, self.multi_size = _get_conf()
        self.field_size = self.single_size + self.numerical_size + self.multi_size
        self.embedding_length = self.field_size * config.embedding_size
        self.config = config
        # Embedding layers
        self.single_first_embedding = tf.keras.layers.Embedding(10000, 1)
        self.numerical_first_embedding = tf.keras.layers.Embedding(10000, 1)
        self.single_second_embedding = tf.keras.layers.Embedding(10000, config.embedding_size)
        self.numerical_second_embedding = tf.keras.layers.Embedding(10000, config.embedding_size)
        # DNN部分
        dnn_net = [self.embedding_length - self.numerical_size * config.embedding_size + self.numerical_size] + config.dnn_net_size
        self.dnn_layers = []
        for i in range(len(config.dnn_net_size)):
            self.dnn_layers.append(tf.keras.layers.Dense(config.dnn_net_size[i], activation='relu'))
            self.dnn_layers.append(tf.keras.layers.BatchNormalization())
        self.dnn_layers = tf.keras.Sequential(self.dnn_layers)
        # 输出层
        output_length = self.field_size + config.embedding_size + config.dnn_net_size[-1]
        self.final_dense = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        single_index, numerical_index, numerical_value, value = inputs
        # 一阶部分
        first_single_result = tf.squeeze(self.single_first_embedding(single_index), axis=-1)
        first_numerical_result = tf.squeeze(self.numerical_first_embedding(numerical_index), axis=-1)
        first_embedding_output = tf.concat([first_single_result, first_numerical_result], axis=1)
        y_first_order = first_embedding_output * value
        # 二阶部分
        second_single_result = tf.reshape(self.single_second_embedding(single_index), [-1, self.single_size * self.config.embedding_size])
        second_numerical_result = tf.reshape(self.numerical_second_embedding(numerical_index), [-1, self.numerical_size * self.config.embedding_size])
        # 修正：dnn_input应包含second_single_result和second_numerical_result
        dnn_input = tf.concat([second_single_result, second_numerical_result], axis=1)
        dnn_input = tf.concat([dnn_input, numerical_value], axis=1)
        dnn_output = self.dnn_layers(dnn_input, training=training)
        # 输出拼接
        output = tf.concat([y_first_order, dnn_output], axis=1)
        logits = self.final_dense(output)
        return tf.nn.softmax(logits)

def get_batch(data, idx, config, single_size, numerical_size, multi_size):
    if idx == -1:
        batch_data = data
    elif (idx + 1) * config.batch_size <= len(data):
        batch_data = data[idx*config.batch_size:(idx+1)*config.batch_size]
    else:
        batch_data = data[idx*config.batch_size:]
    final_label = []
    final_single_index = []
    final_numerical_value = []
    final_numerical_index = []
    final_value = []
    for line in batch_data:
        line_index = []
        line_value = []
        line_numerical_value = []
        line_data = line.split(',')
        final_label.append(int(line_data[0]))
        if single_size:
            for i in range(1, 1 + single_size):
                single_pair = line_data[i].split(':')
                line_index.append(int(single_pair[0]))
                line_value.append(float(single_pair[1]))
        final_single_index.append(line_index)
        line_index = []
        if single_size + numerical_size:
            for i in range(1 + single_size, 1 + single_size + numerical_size):
                single_pair = line_data[i].split(':')
                line_numerical_value.append(float(single_pair[1]))
                line_index.append(int(single_pair[0]))
                line_value.append(float(single_pair[1]))
        final_numerical_value.append(line_numerical_value)
        final_numerical_index.append(line_index)
        final_value.append(line_value)
    return [np.array(final_label), np.array(final_single_index), np.array(final_numerical_index), np.array(final_numerical_value), np.array(final_value)]

def train():
    total_emb, single_size, numerical_size, multi_size = _get_conf()
    train_data = _get_data(Config.train_save_file)
    valid_data = _get_data(Config.valid_save_file)
    model = xDeepFM(Config)
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=Config.learning_rate)
    global_step = 0
    global_train_loss = []
    global_valid_loss = []
    global_steps = []
    for epoch in range(Config.epochs):
        num_batches = int(len(train_data) / Config.batch_size) + 1
        for j in range(num_batches):
            global_step += 1
            batch = get_batch(train_data, j, Config, single_size, numerical_size, multi_size)
            labels = get_label(batch[0], 2)
            with tf.GradientTape() as tape:
                preds = model([batch[1], batch[2], batch[3], batch[4]], training=True)
                loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, preds))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if global_step % 10 == 0:
                valid_batch = get_batch(valid_data, -1, Config, single_size, numerical_size, multi_size)
                valid_labels = get_label(valid_batch[0], 2)
                valid_preds = model([valid_batch[1], valid_batch[2], valid_batch[3], valid_batch[4]], training=False)
                valid_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(valid_labels, valid_preds))
                global_steps.append(global_step)
                global_train_loss.append(loss.numpy())
                global_valid_loss.append(valid_loss.numpy())
                print(f'step: {global_step}, train loss: {loss.numpy()}, valid loss: {valid_loss.numpy()}, valid_auc: {auc_score(valid_preds.numpy(), valid_labels, 2)}')
    pd.DataFrame({'step': global_steps, 'train_loss': global_train_loss, 'valid_loss': global_valid_loss}).to_csv('DeepFM_loss_result_tf2.csv', index=False)

if __name__ == '__main__':
    train()
