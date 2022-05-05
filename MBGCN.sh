#!/bin/bash

# shell file for Tmall
model='MBGCN'
dataset_name='Tmall_release'
gpu_id='5'
create_embeddings='False'
es_patience="50"
embedding_size='64'
lamb='1'

mgnn_weight1='1'
mgnn_weight2='1'
mgnn_weight3='1'
mgnn_weight4='1'

relation='buy,cart,collect,click'

pretrain_path='/data3/jinbowen/multi_behavior/output/Tmall_release/Tmall_release-MF_lr1e-2-L1e-2-size64@jinbowen'

# lr_list=('3e-5')
# L2_list=('1e-5')
# lr_list=('1e-4' '3e-5' '1e-5' '3e-6')
# L2_list=('1e-2' '1e-3' '1e-4' '1e-5' '1e-6')

lr='3e-4'
L2='1e-4'
message_dropout=('0.2')
node_dropout=('0.2')
# message_dropout=('0' '0.1' '0.2' '0.3' '0.4' '0.5')
# node_dropout=('0' '0.1' '0.2' '0.3' '0.4' '0.5')

for md in ${message_dropout[@]}
do
    for nd in ${node_dropout[@]}
    do
        name=${dataset_name}-${model}_lr${lr}-L${L2}-size${embedding_size}-lamb${lamb}-md${md}-nd${nd}@jinbowen
        python main.py \
            --name ${name} \
            --model ${model} \
            --gpu_id ${gpu_id} \
            --dataset_name ${dataset_name} \
            --L2_norm ${L2} \
            --lr ${lr} \
            --create_embeddings ${create_embeddings} \
            --es_patience ${es_patience} \
            --embedding_size ${embedding_size}\
            --mgnn_weight ${mgnn_weight1}\
            --mgnn_weight ${mgnn_weight2}\
            --mgnn_weight ${mgnn_weight3}\
            --mgnn_weight ${mgnn_weight4}\
            --lamb ${lamb} \
            --relation ${relation} \
            --pretrain_path ${pretrain_path}
    done
done
