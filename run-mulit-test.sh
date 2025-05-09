#!/bin/bash

# 設定起始值、結束值和步長
start=0
end=3000
step=100

# 使用 for 迴圈，從 start 到 end，每次增加 step
for ((i=start; i<=end; i+=step))
do
    echo "Executing with checkpoint_${i}.pth"
    # 執行你的 python 命令
    python test.py --checkpoint results/experiment_14/checkpoint_${i}.pth
    
    # 可以在這裡添加一個短暫的等待，如果需要的話
    # sleep 1
done

echo "All runs completed."