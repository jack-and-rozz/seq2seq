#!/bin/bash

#input from shell
usage() {
    echo "Usage:$0 bin_file checkpoint_path mode [config_path]"
    exit 1
}

# optの解析 (argv, argc, opt_names, opt_valuesを設定)
source ./scripts/manually_getopt.sh $@

if [ $argc -lt 2 ];then
    usage;
fi
#bin_file=${argv[0]}
bin_file=bins/wikiP2D.py
checkpoint_path=${argv[0]}
mode=${argv[1]}
config_file=${argv[2]}
config_dir=configs

log_file=$checkpoint_path/${mode}.log
if [ "${config_file}" = "" ]; then
    if [[ "${mode}" =~ "train" ]]; then
	if [ -e ${checkpoint_path}/config ]; then
	    config_file=${checkpoint_path}/config
	#else
	    #echo "specify config file when start training from scratch."
	    #exit 1
	fi
    else 
	config_file=${checkpoint_path}/config
    fi
fi

if [ "${config_file}" != "" ]; then
    . ./${config_file} 
fi

# 実行時オプションを優先(from manually_optget.sh)
for i in $(seq 0 $(expr ${#opt_names[@]} - 1)); do
    name=${opt_names[$i]}
    value=${opt_values[$i]}
    eval $name=$value
done;

# get params from config

params_arr=(mode checkpoint_path log_file w2p_dataset w_vocab_size c_vocab_size batch_size hidden_size learning_rate in_keep_prob out_keep_prob num_layers max_gradient_norm encoder_type c_encoder_type model_type cell_type cbase wbase state_is_tuple max_a_sent_length max_d_sent_length max_a_word_length n_triples graph_task desc_task coref_task)

#num_layers source_lang target_lang source_vocab_size target_vocab_size num_samples hidden_size embedding_size keep_prob seq2seq_type encoder_type decoder_type cell_type batch_size max_epoch max_rows max_sequence_length max_to_keep source_data_dir processed_data_dir vocab_data train_data dev_data test_data num_gpus learning_rate max_gradient_norm init_scale trainable_source_embedding trainable_target_embedding beam_size w2v model_type multi_gpu_wrapper, negative_sampling_rate)
params=""
for param in ${params_arr[@]}; do 
    if [ ! ${!param} = "" ]; then
	params="${params} --${param}=${!param}"
    fi
done;


run(){
    if [ ! -e $checkpoint_path/logs ]; then
	mkdir $checkpoint_path/logs
    fi
    echo "python -B $bin_file $params"
    if [ ! ${output_log} = "" ]; then
	log_file=$checkpoint_path/logs/$(date +'%Y%m%d-%H%M')
	echo 'output_log='$log_file.*
	python -B $bin_file $params > $log_file.log 2>$log_file.err

    else
	python -B $bin_file $params 
    fi

    wait
}
mkdir -p $checkpoint_path
run
wait
