#!/bin/bash

LLAMA_MODELS_DIR=${LLAMA_MODELS_DIR:-}
LLAMA_STACK_DIR=${LLAMA_STACK_DIR:-}
TEST_PYPI_VERSION=${TEST_PYPI_VERSION:-}

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <build_name> <docker_base> <pip_dependencies> [<special_pip_deps>]" >&2
  echo "Example: $0 my-fastapi-app python:3.9-slim 'fastapi uvicorn' " >&2
  exit 1
fi

special_pip_deps="$5"

set -euo pipefail

build_name="$1"
image_name="llamastack-$build_name"
docker_base=$2
build_file_path=$3
pip_dependencies=$4

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPO_DIR=$(dirname $(dirname "$SCRIPT_DIR"))
DOCKER_BINARY=${DOCKER_BINARY:-docker}
DOCKER_OPTS=${DOCKER_OPTS:-}
REPO_CONFIGS_DIR="$REPO_DIR/tmp/configs"

TEMP_DIR=$(mktemp -d)

llama stack configure $build_file_path --output-dir $REPO_CONFIGS_DIR

add_to_docker() {
  local input
  output_file="$TEMP_DIR/Dockerfile"
  if [ -t 0 ]; then
    printf '%s\n' "$1" >>"$output_file"
  else
    # If stdin is not a terminal, read from it (heredoc)
    cat >>"$output_file"
  fi
}

add_to_docker <<EOF
FROM $docker_base
WORKDIR /app

RUN apt-get update && apt-get install -y \
       iputils-ping net-tools iproute2 dnsutils telnet \
       curl wget telnet \
       procps psmisc lsof \
       traceroute \
       bubblewrap \
       python3-pip \
       && rm -rf /var/lib/apt/lists/*

EOF

stack_mount="/app/llama-stack-source"
models_mount="/app/llama-models-source"

# TODO - Otherwise we get an error on pip install using the cuda ubuntu base
# image ... maybe add pip_options to the build template so options can be
# specified along side the custom docker image?
pip_options="--break-system-packages"

if [ -n "$LLAMA_STACK_DIR" ]; then
  if [ ! -d "$LLAMA_STACK_DIR" ]; then
    echo "${RED}Warning: LLAMA_STACK_DIR is set but directory does not exist: $LLAMA_STACK_DIR${NC}" >&2
    exit 1
  fi

  # Install in editable format. We will mount the source code into the container
  # so that changes will be reflected in the container without having to do a
  # rebuild. This is just for development convenience.
  add_to_docker "RUN pip install $pip_options -e $stack_mount"
else
  add_to_docker "RUN pip install $pip_options llama-stack"
fi

if [ -n "$LLAMA_MODELS_DIR" ]; then
  if [ ! -d "$LLAMA_MODELS_DIR" ]; then
    echo "${RED}Warning: LLAMA_MODELS_DIR is set but directory does not exist: $LLAMA_MODELS_DIR${NC}" >&2
    exit 1
  fi

  add_to_docker <<EOF
RUN pip uninstall -y llama-models
RUN pip install $pip_options $models_mount

EOF
fi

if [ -n "$pip_dependencies" ]; then
  add_to_docker "RUN pip install $pip_options $pip_dependencies"
fi

if [ -n "$special_pip_deps" ]; then
  IFS='#' read -ra parts <<< "$special_pip_deps"
  for part in "${parts[@]}"; do
    add_to_docker "RUN pip install $pip_options $part"
  done
fi

add_to_docker <<EOF

# This would be good in production but for debugging flexibility lets not add it right now
# We need a more solid production ready entrypoint.sh anyway
#
ENTRYPOINT ["python3", "-m", "llama_stack.distribution.server.server"]

EOF

add_to_docker "ADD tmp/configs/$(basename "$build_file_path") ./llamastack-build.yaml"
add_to_docker "ADD tmp/configs/$build_name-run.yaml ./llamastack-run.yaml"

printf "Dockerfile created successfully in $TEMP_DIR/Dockerfile"
cat $TEMP_DIR/Dockerfile
printf "\n"

mounts=""
if [ -n "$LLAMA_STACK_DIR" ]; then
  mounts="$mounts -v $(readlink -f $LLAMA_STACK_DIR):$stack_mount"
fi
if [ -n "$LLAMA_MODELS_DIR" ]; then
  mounts="$mounts -v $(readlink -f $LLAMA_MODELS_DIR):$models_mount"
fi

if command -v selinuxenabled &> /dev/null && selinuxenabled; then
  # Disable SELinux labels -- we don't want to relabel the llama-stack source dir
  DOCKER_OPTS="$DOCKER_OPTS --security-opt label=disable"
fi

set -x
$DOCKER_BINARY build $DOCKER_OPTS -t $image_name -f "$TEMP_DIR/Dockerfile" "$REPO_DIR" $mounts
set +x

echo "Success! You can run it with: $DOCKER_BINARY $DOCKER_OPTS run -p 5000:5000 $image_name"
