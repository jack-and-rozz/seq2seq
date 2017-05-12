#!/bin/bash

#input from shell
usage() {
    echo "Usage:$0 model_dir mode [config]"
    exit 1
}

# optの解析 (argv, argc, opt_names, opt_valuesを設定)
source ./scripts/manually_getopt.sh $@
if [ $argc -lt 2 ];then
    usage;
fi

checkpoint_path=${argv[0]}
mode=${argv[1]}
config_file=${argv[2]}
config_dir=configs

log_file=$checkpoint_path/${mode}.log
if [ "${config_file}" = "" ]; then
    if [[ "${mode}" =~ "train" ]]; then
	if [ -e ${checkpoint_path}/config ]; then
	    config_file=${checkpoint_path}/config
	else
	    echo "specify config file when start training from scratch."
	    exit 1
	fi
    else 
	config_file=${checkpoint_path}/config
    fi
fi

. ./${config_file} 

# 実行時オプションを優先(from manually_optget.sh)
for i in $(seq 0 $(expr ${#opt_names[@]} - 1)); do
    name=${opt_names[$i]}
    value=${opt_values[$i]}
    eval $name=$value
done;

# get params from config

params_arr=(checkpoint_path log_file mode num_layers in_vocab_size hidden_size embedding_size encoder_type cell_type batch_size max_train_rows max_to_keep source_data_dir processed_data_dir vocab_data train_data valid_data test_data seq2seq_type cell_type) 
params=""
for param in ${params_arr[@]}; do 
    if [ ! ${!param} = "" ]; then
	params="${params} --${param}=${!param}"
    fi
done;

run(){
    python -B main.py $params
    #python -m cProfile -o profile.stats main.py $params
    wait
}
mkdir -p $checkpoint_path
run
wait