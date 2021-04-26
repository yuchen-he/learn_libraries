#!/bin/bash

# $0 refers to file_name
# $0~$n refers to args
# $# refers to the number of args
echo "$0 is building this project ..."

mkdir build && cd build/
cmake .. -DBUILD_TESTS=OFF
make
