#!/bin/bash

#select-gpu.sh 0 nohup python bins/wikiP2D.py m_wikidata/only_coref train -ct only_coref --cleanup=T >logs/only_coref.log&


target_dir='latest'

#select-gpu.sh 0 nohup python bins/wikiP2D.py m_wikidata/$target_dir/graph.base train -ct only_graph --cleanup=T > logs/graph.base &
#select-gpu.sh 2 nohup python bins/wikiP2D.py m_wikidata/$target_dir/coref.base.h400 train -ct only_coref_h400 --cleanup=T > logs/coref.base & 

#select-gpu.sh 6 nohup python bins/wikiP2D.py m_wikidata/$target_dir/coref.base train -ct only_coref --cleanup=T >logs/coref.base.log 2>logs/coref.base.err& 
#select-gpu.sh 7 nohup python bins/wikiP2D.py m_wikidata/$target_dir/category.base train -ct only_category --cleanup=T >logs/category.base.log 2>logs/category.base.err& 

select-gpu.sh 6 nohup python bins/wikiP2D.py m_wikidata/$target_dir/category.base.all train -ct only_category --cleanup=T >logs/category.base.all.log 2>logs/category.base.all.err& 

