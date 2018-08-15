#!/bin/bash



usage() {
    echo "Usage:$0 config_type gpu_ids"
    exit 1
}
if [ $# -lt 2 ];then
    usage;
fi

if [ $# -lt 3 ];then
    target_dir='latest'
else
    target_dir= $3
fi
config_type=$1
gpu_ids=$2
select-gpu.sh $gpu_ids nohup python bins/wikiP2D.py m_wikidata/$target_dir/$config_type train -ct $config_type --cleanup=T > logs/$config_type.log 2>logs/$config_type.err &
#select-gpu.sh 2,3 nohup python bins/wikiP2D.py m_wikidata/$target_dir/mtl.meanloss train -ct mtl_meanloss --cleanup=T > logs/mtl.meanloss &

#select-gpu.sh 4,5 nohup python bins/wikiP2D.py m_wikidata/$target_dir/mtl.adv train -ct mtl_adv --cleanup=T > logs/mtl.adv &

#select-gpu.sh 6,7 nohup python bins/wikiP2D.py m_wikidata/$target_dir/mtl.onebyone train -ct mtl_onebyone --cleanup=T > logs/mtl.onebyone &

#select-gpu.sh 6,7 nohup python bins/wikiP2D.py m_wikidata/$target_dir/mtl.adv.local train -ct mtl_adv_local --cleanup=T > logs/mtl.adv.local &




