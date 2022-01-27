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

import tensorflow as tf
import numpy as np

from architectures import Additive_LSTM_Demo, Additive_LSTM, Additive_GRU, Oneshot_LSTM
from configs import additive_lstm_demo_cfg, additive_lstm_cfg, additive_gru_cfg, oneshot_lstm_cfg

model_map = {
    "additive_lstm_demo": Additive_LSTM_Demo.LidarCompressionNetwork,
    "additive_lstm": Additive_LSTM.LidarCompressionNetwork,
    "additive_gru": Additive_GRU.LidarCompressionNetwork,
    "oneshot_lstm": Oneshot_LSTM.LidarCompressionNetwork
}

config_map = {
    "additive_lstm_demo": additive_lstm_demo_cfg.additive_lstm_demo_cfg,
    "additive_lstm": additive_lstm_cfg.additive_lstm_cfg,
    "additive_gru": additive_gru_cfg.additive_gru_cfg,
    "onshot_lstm": oneshot_lstm_cfg.oneshot_lstm_cfg,
}


def load_config_and_model(args):
    cfg = config_map[args.model.lower()]()

    # overwrite default values in config with parsed arguments
    for key, value in vars(args).items():
        if value:
            cfg[key] = value

    # support multi GPU training
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = model_map[args.model.lower()](
            bottleneck=cfg.bottleneck,
            num_iters=cfg.num_iters,
            batch_size=cfg.batch_size,
            input_size=cfg.crop_size
        )
        # init model
        model(np.random.random((
            cfg.batch_size,
            cfg.crop_size,
            cfg.crop_size,
            1))
        )
    model.summary()

    # set batch size for train and val data loader
    cfg.train_data_loader_batch_size = cfg.batch_size * mirrored_strategy.num_replicas_in_sync
    cfg.val_data_loader_batch_size = cfg.val_batch_size * mirrored_strategy.num_replicas_in_sync
    return cfg, model
