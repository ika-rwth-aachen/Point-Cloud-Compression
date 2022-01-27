#!/bin/bash

#
#  ==============================================================================
#  MIT License
#
#  Copyright 2022 Institute for Automotive Engineering of RWTH Aachen University.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ==============================================================================
#

# get directory of this script
DIR="$(cd -P "$(dirname "$0")" && pwd)"

RED='\033[0;31m'
NC='\033[0m' # No Color

export GPG_TTY=$(tty) # set tty for login at docker container registry using credentials helper 'pass'

IMAGE_NAME="tillbeemelmanns/pointcloud_compression" \
IMAGE_TAG="noetic" \
MOUNT_DIR="$DIR/../catkin_ws" \
DOCKER_MOUNT_DIR="/catkin_ws" \
CONTAINER_NAME="ros_compression_container" \
DOCKER_RUN_ARGS="--workdir /catkin_ws" \
\
$DIR/run-ros.sh $@