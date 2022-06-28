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

RED='\033[0;31m'
NC='\033[0m'
DIR="$(cd -P "$(dirname "$0")" && pwd)"  # path to directory containing this script

function help_text() {
  cat << EOF
    usage: run-ros.sh [-a][-d][-g gpu_ids][-h][-i][-n name] [COMMAND]
      -a             directly attach to a container shell
      -d             disable X11-GUI-forwarding to the host's X server
      -g             specify comma-separated IDs of GPUs to be available in the container; defaults to 'all'; use 'none' in case of non-NVIDIA GPUs
      -h             show help text
      -n             container name
EOF
  exit 0
}

# PARAMETERS can be passed as environment variables #
if [[ -z "${IMAGE_NAME}" ]]; then
  IMAGE_NAME="tillbeemelmanns/pointcloud_compression:noetic"
fi
if [[ -z "${IMAGE_TAG}" ]]; then
  IMAGE_TAG="latest"
fi
if [[ -z "${DOCKER_USER}" ]]; then
  DOCKER_USER="rosuser"
fi
DOCKER_HOME="/home/${DOCKER_USER}"


if [[ -z "${CONTAINER_NAME}" ]]; then
  CONTAINER_NAME="ros"  # name of the Docker container to be started
fi
if [[ -z "${MOUNT_DIR}" ]]; then
  MOUNT_DIR="$DIR"  # host's directory to be mounted into the container
fi
if [[ -z "${DOCKER_MOUNT_DIR}" ]]; then
  DOCKER_MOUNT_DIR="${DOCKER_HOME}/ws"  # container directory that host directory is mounted into
fi
if ! [[ -z "${DOCKER_MOUNT}" ]]; then
  echo -e "${RED}The DOCKER_MOUNT argument is deprecated. The folder $(readlink -f ${MOUNT_DIR}) will be mounted to ${DOCKER_MOUNT_DIR}.${NC}"
fi
if [[ -z "${DOCKER_RUN_ARGS}" ]]; then
  DOCKER_RUN_ARGS=()  # additional arguments for 'docker run'
else
  DOCKER_RUN_ARGS=(${DOCKER_RUN_ARGS})
fi
if [[ -z ${BASHRC_COMMANDS} ]]; then
  unset BASHRC_COMMANDS
fi
# PARAMETERS END #

IMAGE="$IMAGE_NAME:$IMAGE_TAG"
DIR="$(cd -P "$(dirname "$0")" && pwd)"  # absolute path to the directory containing this script

# defaults
ATTACH="0"
HAS_COMMAND="0"
X11="1"
GPU_IDS="all"

# read command line arguments
while getopts ac:dg:hin:oru opt
do
   case $opt in
       a) ATTACH="1";;
       d) echo "Headless Mode: no X11-GUI-forwarding"; X11="0";;
       g) GPU_IDS="$OPTARG";;
       h) echo "Help: "; help_text;;
       n) CONTAINER_NAME="$OPTARG";;
   esac
done
shift $(($OPTIND - 1))
if ! [[ -z ${@} ]]; then
  HAS_COMMAND="1"
fi


# create custom .bashrc appendix, if startup commands are given
if [[ ! -z ${BASHRC_COMMANDS} ]]; then
  BASHRC_APPENDIX=${DIR}/.bashrc-appendix
  DOCKER_BASHRC_APPENDIX=${DOCKER_HOME}/.bashrc-appendix
  :> ${BASHRC_APPENDIX}
  echo "${BASHRC_COMMANDS}" >> ${BASHRC_APPENDIX}
fi

# try to find running container to attach to
RUNNING_CONTAINER_NAME=$(docker ps --format '{{.Names}}' | grep -x "${CONTAINER_NAME}" | head -n 1)
if [[ -z ${RUNNING_CONTAINER_NAME} ]]; then # look for containers of same image if provided name is not found
  RUNNING_CONTAINER_NAME=$(docker ps --format '{{.Image}} {{.Names}}' | grep ${IMAGE} | head -n 1 | awk '{print $2}')
  if [[ ! -z ${RUNNING_CONTAINER_NAME} ]]; then
    read -p "Found running container '${RUNNING_CONTAINER_NAME}', would you like to attach instead of launching a new container '${CONTAINER_NAME}'? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      RUNNING_CONTAINER_NAME=""
    fi
  fi
fi


if [[ ! -z ${RUNNING_CONTAINER_NAME} ]]; then
  # attach to running container

  # collect all arguments for docker exec
  DOCKER_EXEC_ARGS+=("--interactive")
  DOCKER_EXEC_ARGS+=("--tty")
  DOCKER_EXEC_ARGS+=("--user ${DOCKER_USER}")
  if [[ ${X11} = "0" ]]; then
    DOCKER_EXEC_ARGS+=("--env DISPLAY=""")
  fi

  # attach
  echo "Attaching to running container ..."
  echo "  Name: ${RUNNING_CONTAINER_NAME}"
  echo "  Image: ${IMAGE}"
  echo ""
  if [[ -z ${@} ]]; then
    docker exec ${DOCKER_EXEC_ARGS[@]} ${RUNNING_CONTAINER_NAME} bash -i
  else
    docker exec ${DOCKER_EXEC_ARGS[@]} ${RUNNING_CONTAINER_NAME} bash -i -c "${*}"
  fi

else
  # start new container
  ROS_MASTER_PORT=11311

  # GUI Forwarding to host's X server [https://stackoverflow.com/a/48235281]
  # Uncomment "X11UseLocalhost no" in /etc/ssh/sshd_config to enable X11 forwarding through SSH for isolated containers
  if [ $X11 = "1" ]; then
    XSOCK=/tmp/.X11-unix  # to support X11 forwarding in isolated containers on local host (not thorugh SSH)
    XAUTH=$(mktemp /tmp/.docker.xauth.XXXXXXXXX)
    xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
    chmod 777 $XAUTH
  fi

  # set --gpus flag for docker run
  if ! [[ "$GPU_IDS" == "all" || "$GPU_IDS" == "none" ]]; then
    GPU_IDS=$(echo "\"device=$GPU_IDS\"")
  fi
  echo "GPUs: $GPU_IDS"
  echo ""

  # collect all arguments for docker run
  DOCKER_RUN_ARGS+=("--name ${CONTAINER_NAME}")
  DOCKER_RUN_ARGS+=("--rm")
  if ! [[ "$GPU_IDS" == "none" ]]; then
    DOCKER_RUN_ARGS+=("--gpus ${GPU_IDS}")
  fi
  DOCKER_RUN_ARGS+=("--env TZ=$(cat /etc/timezone)")
  DOCKER_RUN_ARGS+=("--volume ${MOUNT_DIR}:${DOCKER_MOUNT_DIR}")
  DOCKER_RUN_ARGS+=("--env DOCKER_USER=${DOCKER_USER}")
  DOCKER_RUN_ARGS+=("--env DOCKER_HOME=${DOCKER_HOME}")
  DOCKER_RUN_ARGS+=("--env DOCKER_MOUNT_DIR=${DOCKER_MOUNT_DIR}")

  if [[ ${ATTACH} = "1" &&  ${HAS_COMMAND} = "0" ]]; then
    DOCKER_RUN_ARGS+=("--detach")
    DOCKER_EXEC_ARGS+=("--interactive")
    DOCKER_EXEC_ARGS+=("--tty")
    DOCKER_EXEC_ARGS+=("--user ${DOCKER_USER}")
  fi
  if [[ ${HAS_COMMAND} = "1" ]]; then
    DOCKER_RUN_ARGS+=("--interactive")
    DOCKER_RUN_ARGS+=("--tty")
  fi

  DOCKER_RUN_ARGS+=("--network host")
  DOCKER_RUN_ARGS+=("--env ROS_MASTER_URI=http://$(hostname):${ROS_MASTER_PORT}")

  if [[ ${X11} = "1" ]]; then
    DOCKER_RUN_ARGS+=("--env QT_X11_NO_MITSHM=1")
    DOCKER_RUN_ARGS+=("--env DISPLAY=${DISPLAY}")
    DOCKER_RUN_ARGS+=("--env XAUTHORITY=${XAUTH}")
    DOCKER_RUN_ARGS+=("--volume ${XAUTH}:${XAUTH}")
    DOCKER_RUN_ARGS+=("--volume ${XSOCK}:${XSOCK}")
  fi
  if [[ ! -z ${BASHRC_COMMANDS} ]]; then
    DOCKER_RUN_ARGS+=("--volume ${BASHRC_APPENDIX}:${DOCKER_BASHRC_APPENDIX}:ro")
  fi

  # run docker container
  echo "Starting container ..."
  echo "  Name: ${CONTAINER_NAME}"
  echo "  Image: ${IMAGE}"
  echo ""
  if [[ ${HAS_COMMAND} = "0" ]]; then
    docker run ${DOCKER_RUN_ARGS[@]} ${IMAGE}
    if [[ ${ATTACH} = "1" ]]; then
      echo
      docker exec ${DOCKER_EXEC_ARGS[@]} ${CONTAINER_NAME} bash -i
    fi
  else
    docker run ${DOCKER_RUN_ARGS[@]} ${IMAGE} bash -i -c "${*}"
  fi

  # cleanup
  if [ $X11 = "1" ]; then
    rm $XAUTH
  fi
  if [[ ! -z ${BASHRC_COMMANDS} ]]; then
    rm ${BASHRC_APPENDIX}
  fi
fi

if [[ -z ${RUNNING_CONTAINER_NAME} ]]; then
  RUNNING_CONTAINER_NAME=$(docker ps --format '{{.Names}}' | grep "\b${CONTAINER_NAME}\b")
fi
echo
if [[ ! -z ${RUNNING_CONTAINER_NAME} ]]; then
  echo "Container '${RUNNING_CONTAINER_NAME}' is still running: can be stopped via 'docker stop ${RUNNING_CONTAINER_NAME}'"
else
  echo "Container '${CONTAINER_NAME}' has stopped"
fi
