#!/bin/bash
target_dir=latest
file_sizes=(10k 100k)
mask_types=(mask nomask)

#nohup python bins/wikiP2D.py m_wikidata/$target_dir/only_coref c_test &
for file_size in ${file_sizes[@]}; do
    for mask_type in ${mask_types[@]}; do
	nohup python bins/wikiP2D.py m_wikidata/$target_dir/only_graph.${mask_type}.${file_size} g_test &
	nohup python bins/wikiP2D.py m_wikidata/$target_dir/mtl.${mask_type}.${file_size} g_test &
	nohup python bins/wikiP2D.py m_wikidata/$target_dir/mtl.${mask_type}.${file_size} c_test &
    done;
done;

#suffix=.10k
#nohup python bins/wikiP2D.py m_wikidata/$target_dir/only_graph.nomask${suffix} g_test &
#nohup python bins/wikiP2D.py m_wikidata/$target_dir/only_graph.mask${suffix} g_test &
#nohup python bins/wikiP2D.py m_wikidata/$target_dir/mtl.mask${suffix} c_test &
#nohup python bins/wikiP2D.py m_wikidata/$target_dir/mtl.mask${suffix} g_test &
#nohup python bins/wikiP2D.py m_wikidata/$target_dir/mtl.nomask${suffix} c_test &
#nohup python bins/wikiP2D.py m_wikidata/$target_dir/mtl.nomask${suffix} g_test &
