#!/bin/bash


target_dir='latest'
#select-gpu.sh 0,1 nohup python bins/wikiP2D.py m_wikidata/$target_dir/mtl.iterative train -ct mtl_iterative --cleanup=T > logs/mtl.iterative &
#select-gpu.sh 2,3 nohup python bins/wikiP2D.py m_wikidata/$target_dir/mtl.meanloss train -ct mtl_meanloss --cleanup=T > logs/mtl.meanloss &

#select-gpu.sh 4,5 nohup python bins/wikiP2D.py m_wikidata/$target_dir/mtl.adv train -ct mtl_adv --cleanup=T > logs/mtl.adv &

#select-gpu.sh 6,7 nohup python bins/wikiP2D.py m_wikidata/$target_dir/mtl.onebyone train -ct mtl_onebyone --cleanup=T > logs/mtl.onebyone &

select-gpu.sh 4,5,6 nohup python bins/wikiP2D.py m_wikidata/$target_dir/mtl.adv.local train -ct mtl_adv_local --cleanup=T > logs/mtl.adv.local &




