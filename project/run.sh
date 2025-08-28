#!/bin/bash

srun --partition aps --gres=gpu:4 --exclusive ./main $@
