#!/bin/sh

DATA_DIR="./lfw_funneled/"


if [ ! -d "$DATA_DIR" ]; then
    echo "Download Data"
    wget http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
    tar -xvzf lfw-funneled.tgz
    rm lfw-funneled
fi
