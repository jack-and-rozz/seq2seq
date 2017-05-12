#!/bin/bash
 #自分でオプションを解析。順番自由、ロングオプションも可能
# http://qiita.com/b4b4r07/items/dcd6be0bb9c9185475bb

declare -i argc=0
declare -a argv=()

opt_names=()
opt_values=()

while (( $# > 0 ))
do
    case "$1" in
        --*)
            if [[ "$1" =~ ^--(.+)=(.+)$ ]]; then
		opt_names=("${opt_names[@]}" ${BASH_REMATCH[1]})
		opt_values=("${opt_values[@]}" ${BASH_REMATCH[2]})
            fi
            shift
            ;;

        -*)
            if [[ "$1" =~ 'n' ]]; then
                nflag='-n'
            fi
            shift
            ;;
        *)
            ((++argc))
            argv=("${argv[@]}" "$1")
            shift
            ;;
    esac
done

for i in $(seq 0 $(expr ${#opt_names[@]} - 1)); do
    name=${opt_names[$i]}
    value=${opt_values[$i]}
    eval $name=$value
done;


#echo ${opt_names[@]}
#echo ${opt_values[@]}
#echo 'argc='$argc
#echo 'argv='${argv[@]}

#$ echo ${arr[0]}   # インデックス0の値
#$ echo $arr   # 先頭の値のみ
#$ echo ${arr[@]}   # 全ての値
