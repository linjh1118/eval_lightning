#!/bin/bash

# get the file path of the script
script_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd $script_path/../..
echo "current path: $(pwd)"


# 设置Wandb环境变量
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=cf0164e1018e601530a4b2c4eb70e44a8ed81945
wandb login --relogin ${WANDB_API_KEY}

# 运行示例脚本
python examples/eval_with_wandb.py



