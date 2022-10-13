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

from easydict import EasyDict


def additive_lstm_cfg():
    """Configuration for the Additive LSTM Framework"""
    cfg = EasyDict()

    # Data loader
    cfg.train_data_dir = "demo_samples/training"
    cfg.val_data_dir = "demo_samples/validation"

    # Training
    cfg.epochs = 3000
    cfg.batch_size = 128
    cfg.val_batch_size = 128
    cfg.save_freq = 10000
    cfg.train_output_dir = "output"
    cfg.xla = True
    cfg.mixed_precision = False

    # Learning Rate scheduler
    cfg.lr_init = 1e-4
    cfg.min_learning_rate = 5e-7
    cfg.min_learning_rate_epoch = cfg.epochs
    cfg.max_learning_rate_epoch = 0

    # Network architecture
    cfg.bottleneck = 32
    cfg.num_iters = 32
    cfg.crop_size = 32

    # Give path for checkpoint or set to False otherwise
    cfg.checkpoint = False
    return cfg
