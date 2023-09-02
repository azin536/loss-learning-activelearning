import random
import tensorflow as tf
import typing
from tqdm import tqdm

from keras.utils.vis_utils import plot_model

from .base import ModelBuilderBase

tfk = tf.keras
tfkl = tfk.layers
K = tfk.backend


class ResnetBlock(tfk.Model):
    """
    A standard resnet block.
    """
    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = tfkl.Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = tfkl.BatchNormalization()
        self.conv_2 = tfkl.Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = tfkl.BatchNormalization()
        self.merge = tfkl.Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = tfkl.Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = tfkl.BatchNormalization()

    def call(self, inputs):
        res = inputs
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)
        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18(tfk.Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = tfkl.Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = tfkl.BatchNormalization()
        self.pool_2 = tfkl.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = tfkl.GlobalAveragePooling2D()
        self.flat = tfkl.Flatten()
        self.fc = tfkl.Dense(num_classes, activation="softmax")
        self.features = list()

    def get_model(self, input_layer):
        x = self.conv_1(input_layer)
        x = self.init_bn(x)
        x = tf.nn.relu(x)
        x = self.pool_2(x)
        for i, res_block in enumerate([self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]):
            x = res_block(x)
            if i % 2 == 1:
                self.features.append(x)
        x = self.avg_pool(x)
        x = self.flat(x)
        out = self.fc(x)
        # model = tfk.Model(inputs=input_layer, outputs=out, name='target_model')
        # tf.keras.utils.plot_model(model, to_file='./model_target.png')
        return out, self.features

class LossNet:
    def get_loss_pred_model(self, features):
        inputs = tfkl.Input(shape=(1))
        out1 = tfkl.GlobalAveragePooling2D()(features[0])
        out1 = tfkl.Dense(128, activation='relu')(out1)

        out2 = tfkl.GlobalAveragePooling2D()(features[1])
        out2 = tfkl.Dense(128, activation='relu')(out2)

        out3 = tfkl.GlobalAveragePooling2D()(features[2])
        out3 = tfkl.Dense(128, activation='relu')(out3)

        out4 = tfkl.GlobalAveragePooling2D()(features[3])
        out4 = tfkl.Dense(128, activation='relu')(out4)

        out_ = tfkl.Concatenate(axis=1)([out1, out2, out3, out4])
        out = tfkl.Dense(1, activation='relu', name='loss_prediction_module')(out_)
        return out


class LossLearningModel(ModelBuilderBase):

    def __init__(self, config):
        super().__init__(config=config)
        self.epoch = 0
        self.batch_size = self.config.data_pipeline.batch_size
        self.ratio = 1 / self.batch_size
        mb_conf = self.config.model_builder
        self.input_shape = mb_conf.input_shape
        self.n_task = len(self.config.target_labels)
        self.lr = mb_conf.learning_rate
        self.momentum = mb_conf.momentum
        self.weight_decay = mb_conf.decay

    def _get_pred_loss(self, target_loss, pred_loss):
        # target_loss = target_loss[0]
        # print(target_loss)
        # print(pred_loss)
        margin = self.config.loss_learning.margin
        pred = (pred_loss - tf.experimental.numpy.flip(pred_loss, 0))[:len(pred_loss)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target_loss - tf.experimental.numpy.flip(target_loss, 0))[:len(target_loss)//2]
        one = 2 * tf.experimental.numpy.sign(tf.experimental.numpy.clip(target, a_min=0, a_max=None)) - 1
        # print(one)
        # print(target)
        # print("...................")
        # Note that the size of input is already halved
        loss = tf.experimental.numpy.sum(tf.experimental.numpy.clip(margin - (one * pred), a_min=0, a_max=None))
        return loss

    def _get_epoch(self, epoch):
        self.epoch = epoch
        return self.epoch

    # def _get_custom_loss(self, y_true, y_pred):
    #     y_pred = tf.experimental.numpy.asarray(y_pred)
    #     y_pred_target = y_pred[0]
    #     print(y_pred_target.shape)
    #     import numpy as np
    #     y_pred_target = np.reshape(y_pred_target.shape[1], y_pred_target.shape[0])
    #     target_loss = K.binary_crossentropy(y_true, y_pred_target)
    #     print("++++++++++++++++++++++")
    #     # print(tf.experimental.numpy.asarray(y_pred_target))
    #     # print(tf.experimental.numpy.asarray(y_pred_loss))
    #     pred_loss_loss = self._get_pred_loss(target_loss, y_pred[1][0])
    #     if self.epoch > self.config.loss_learning.loss_epoch:
    #         pred_loss_loss
    #     return self.ratio * (target_loss + (self.config.loss_learning.lambda_ * pred_loss_loss))

    def get_compiled_model(self):
        inputs = tfkl.Input(shape=self.input_shape)
        float_inputs = tfkl.Lambda(lambda x: x / 255.0)(inputs)

        resnet = ResNet18(self.n_task)
        resnet_output, features = resnet.get_model(float_inputs)
        # resnet_output = resnet_model(float_inputs)

        loss_net = LossNet()
        loss_module_output = loss_net.get_loss_pred_model(features)

        # stacked_outputs = tf.stack((resnet_output, loss_module_output), axis=0)
        # concatted_outputs = tf.keras.layers.Concatenate(axis=0)([resnet_output, loss_module_output])
        # breakpoint()
        whole_model = tfk.Model(inputs=float_inputs, outputs=[resnet_output, loss_module_output])

        optimizer = tfk.optimizers.legacy.Adam(learning_rate=self.lr)
        metrics = [
            tfk.metrics.SensitivityAtSpecificity(0.8),
            tfk.metrics.AUC(curve="PR", name="AUC of Precision-Recall Curve"),
            tfk.metrics.FalseNegatives(),
            tfk.metrics.FalsePositives(),
            tfk.metrics.TrueNegatives(),
            tfk.metrics.TruePositives(),
        ]
        # whole_model.compile(optimizer=optimizer, loss=self._get_custom_loss,
        #                     metrics=metrics)
        return whole_model

    def train(self, train_dataset, model):
        import time

        optimizer = tfk.optimizers.legacy.Adam(learning_rate=self.lr)
        train_sen_pec = tfk.metrics.SensitivityAtSpecificity(0.8)
        train_auc = tfk.metrics.AUC(curve="PR", name="AUC of Precision-Recall Curve")
        train_fn = tfk.metrics.FalseNegatives()
        train_fp = tfk.metrics.FalsePositives()
        train_tn = tfk.metrics.TrueNegatives()
        train_tp = tfk.metrics.TruePositives()
        metrics = [
            train_sen_pec, train_auc, train_fn, train_fp, train_tn, train_tp
        ]
        epochs = 10
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(tqdm(train_dataset)):
                loss_values = list()
                # print(y_batch_train)
                with tf.GradientTape() as tape:
                    logits = model(x_batch_train, training=True)
                    loss = K.binary_crossentropy(y_batch_train[0], logits[0])
                #     for y_batch, logit in zip(y_batch_train, logits[0]):
                #         loss_values.append(K.binary_crossentropy(y_batch, logit))
                    pred_loss_loss = self._get_pred_loss(loss, logits[1])
                    final_loss = ((1 / len(x_batch_train)) * sum(loss)) + ((2 / len(x_batch_train)) * pred_loss_loss)
                    # final_loss = final_loss[0]
                # final_loss = tf.repeat([[final_loss/13]], 4, axis=1)[0]
                # grads = tape.gradient(final_loss, model.trainable_weights)
                grads = tape.gradient(final_loss, model.trainable_weights)
                # print(grads)
                # breakpoint()
                tf.keras.utils.plot_model(model, to_file='./model_arch.png')
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Update training metric.
                # print(y_batch_train)
                # print(logits[0])
                for m in metrics:
                    m.update_state(y_batch_train[0], logits[0])

                # Log every 200 batches.
                # if step % 5 == 0:
                #     print(
                #         "Training loss (for one batch) at step %d: %.4f"
                #         % (step, float(final_loss))
                #     )
                #     # print("Seen so far: %d samples" % ((step + 1) * len(x_batch_train)))

            # Display metrics at the end of each epoch.
            train_metrics_results = []
            for m in metrics:
                train_metrics_results.append(m.result())
                print(m)

                # Reset training metrics at the end of each epoch
                m.reset_states()

            # Run a validation loop at the end of each epoch.
            # for x_batch_val, y_batch_val in val_dataset:
            #     val_logits = model(x_batch_val, training=False)
            #     # Update val metrics
            #     val_acc_metric.update_state(y_batch_val, val_logits)
            # val_acc = val_acc_metric.result()
            # val_acc_metric.reset_states()
            # print("Validation acc: %.4f" % (float(val_acc),))
            # print("Time taken: %.2fs" % (time.time() - start_time))
