#! /bin/bash
# bash 脚本用于解析nvidia-smi 命令
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
#echo "$GPU_COUNT"
echo "GPU count  is $GPU_COUNT"
TEMP0=$(nvidia-smi -q -i 0 | grep -i "GPU Current Temp")
TEMP1=$(nvidia-smi -q -i 1 | grep -i "GPU Current Temp")
MEM0=$(nvidia-smi -i 0 --query-gpu=utilization.memory --format=csv,noheader)
MEM1=$(nvidia-smi -i 1 --query-gpu=utilization.memory --format=csv,noheader)
PERCENT0=$(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader)
PERCENT1=$(nvidia-smi -i 1 --query-gpu=utilization.gpu --format=csv,noheader)
echo "GPU0 :$TEMP0 MEM : $MEM0"
echo "GPU1 :$TEMP1 MEM : $MEM1"
echo "GPU0 使用率为 $PERCENT0"
echo "GPU1 使用率为 $PERCENT1"
# set python path according to your actual environment
pythonpath='python'
let MAX=$((30))
if [[ "$PERCENT0" -gt "30" ]];then
    echo "GPU0 使用率为 $PERCENT0"
fi

# printf "ID: %-5s  NAME: %-20s  Avage: %4.2f\n" GPU0  tom 74.23333;printf "ID: %-5s  NAME: %-20s  Avage: %4.2f\n" GPU1  jack 81.66666;