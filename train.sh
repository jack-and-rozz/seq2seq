#!/bin/bash



usage() {
    echo "Usage:$0 target_path config_type gpu_ids [cleanup (T or F)]"
    exit 1
}
if [ $# -lt 3 ];then
    usage;
fi

target_path=$1
config_type=$2
gpu_ids=$3
cleanup=$4
if [ $# -lt 4 ]; then
  cleanup=T
fi

select-gpu.sh $gpu_ids nohup python bins/wikiP2D.py $target_path train -ct $config_type --cleanup=$cleanup --log_device_placement=T >> logs/$config_type.log 2>>logs/$config_type.err &


