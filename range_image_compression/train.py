#  ==============================================================================
#  MIT License
#  #
#  Copyright 2022 Institute for Automotive Engineering of RWTH Aachen University.
#  #
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  #
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#  #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ==============================================================================

import os
import argparse

import tensorflow as tf
from keras.callbacks import ModelCheckpoint

from utils import load_config_and_model
from data import LidarCompressionData

from callbacks import TensorBoard, CosineLearningRateScheduler


def train(args):
    cfg, model = load_config_and_model(args)

    train_set = LidarCompressionData(
        input_dir=cfg.train_data_dir,
        crop_size=cfg.crop_size,
        batch_size=cfg.train_data_loader_batch_size,
        augment=True
    ).build_dataset()

    val_set = LidarCompressionData(
        input_dir=cfg.val_data_dir,
        crop_size=cfg.crop_size,
        batch_size=cfg.val_data_loader_batch_size,
        augment=False
    ).build_dataset()

    lr_schedule = CosineLearningRateScheduler(
        cfg.lr_init,
        min_learning_rate=cfg.min_learning_rate,
        min_learning_rate_epoch=cfg.min_learning_rate_epoch,
        max_learning_rate_epoch=cfg.max_learning_rate_epoch
    ).get_callback()

    optimizer = tf.keras.optimizers.Adam()

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(cfg.train_output_dir, f"weights_e={{epoch:05d}}.hdf5"),
        save_freq=cfg.save_freq,
        save_weights_only=True,
        monitor='val_loss',
    )

    tensorboard_callback = TensorBoard(cfg.train_output_dir, val_set, cfg.batch_size)

    if cfg.checkpoint:
        model.load_weights(cfg.checkpoint, by_name=True)
        print("Checkpoint ", cfg.checkpoint, " loaded.")

    model.compile(optimizer=optimizer)

    model.fit(
        train_set,
        validation_data=val_set,
        callbacks=[checkpoint_callback, tensorboard_callback, lr_schedule],
        epochs=cfg.epochs
    )

    model.save_weights(filepath=os.path.join(args.train_output_dir, 'final_model.hdf5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse Flags for the training script!')
    parser.add_argument('-t', '--train_data_dir', type=str,
                        help='Absolute path to the train dataset')
    parser.add_argument('-v', '--val_data_dir', type=str,
                        help='Absolute path to the validation dataset')
    parser.add_argument('-e', '--epochs', type=int,
                        help='Maximal number of training epochs')
    parser.add_argument('-o', '--train_output_dir', type=str,
                        help="Directory where to write the Tensorboard logs and checkpoints")
    parser.add_argument('-s', '--save_freq', type=int,
                        help="Save freq for keras.callbacks.ModelCheckpoint")
    parser.add_argument('-m', '--model', type=str, default='additive_lstm_demo',
                        help='Model name either `additive_gru`, `additive_lstm`,'
                             ' `additive_lstm_demo`, `oneshot_lstm`')
    args = parser.parse_args()
    train(args)
