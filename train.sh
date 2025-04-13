#!/bin/sh


# determine the shell type
if [ -n "$FISH_VERSION" ]; then
    echo "Running in Fish shell"
    set -x PYTHONPATH $PYTHONPATH (pwd)
else
    echo "Running in Bash shell"
    export PYTHONPATH=$PYTHONPATH:$(pwd)
fi

# check the task to perform
if [ "$1" = "clean" ]; then
    # preprocess the data
    echo "Starting data cleaning..."
    python utils/clean_csv.py
elif [ "$1" = "preprocess" ]; then
    # preprocess the data
    echo "Starting data preprocessing..."
    python utils/clean_csv.py
    python preprocessing/pre_data.py
    python preprocessing/get_droo_input_data.py
elif [ "$1" = "train" ]; then
    # train the model
    echo "Starting model training..."
    python train.py
else
    echo "Usage: $0 [preprocess|train]"
    exit 1
fi