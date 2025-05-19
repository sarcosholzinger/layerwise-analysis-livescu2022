model_name=$1
ckpt_dir=$2
data_sample=$3
rep_type=$4
span=$5
subset_id=$6
dataset_split=$7
save_dir_pth=$8
pckg_dir=$9
dataset=$10

model_type=pretrained

if [[ $model_name == "avhubert"* ]]; then
	pckg_dir="$pckg_dir/av_hubert/avhubert"
elif [[ $model_name == "wavlm"* ]]; then
	pckg_dir="$pckg_dir/unilm/wavlm"
elif [[ $model_name == "fastvgs"* ]]; then
	pckg_dir="$pckg_dir/FaST-VGS-Family"
elif [[ $model_name == "xlsr"* ]]; then
	pckg_dir="$pckg_dir"
else
	pckg_dir="$pckg_dir"
fi

if [[ $model_name == "fastvgs"* ]]; then
	ckpt_pth="$ckpt_dir/${model_name}"
else
	ckpt_pth="$ckpt_dir/${model_name}.pt"
fi

offset=False
if [ "$span" = "frame" ]; then
	if [ "$dataset" = "librispeech" ]; then
		utt_id_fn="data_samples/${dataset}/frame_level/500_ids_sample${data_sample}_${dataset_split}.tsv"
	elif [ "$dataset" = "buckeye" ]; then
		utt_id_fn="data_samples/${dataset}/segmentation/${dataset}_${dataset_split}.tsv"
	fi
	mean_pooling=False
	save_dir="${save_dir_pth}/${model_name}/${dataset}_${dataset_split}_sample${data_sample}/${rep_type}/${span}_level"
elif [ "$span" = "all_words" ]; then
	mean_pooling=True
	utt_id_fn="data_samples/${dataset}/${span}_level/all_words_200instances/word_segments_${subset_id}.lst"
	save_dir="${save_dir_pth}/${model_name}/${rep_type}/${span}_level/${dataset}_all_words_200instances/${subset_id}"
else
	mean_pooling=True
	utt_id_fn="data_samples/${dataset}/${span}_level/${dataset_split}_segments_sample${data_sample}_${subset_id}.json"
	if [ "$span" = "phone" ]; then
		offset=True
	fi
	save_dir="${save_dir_pth}/${model_name}/${dataset}_${dataset_split}_sample${data_sample}/${rep_type}/${span}_level/${subset_id}"
fi

echo "Removing any existing features extracted for sample ${data_sample}"
if [ -f $save_dir ]; then
	rm -r $save_dir
fi

fbank_dir="${save_dir_pth}/fbanks/${dataset}_${dataset_split}_sample${data_sample}/"
mkdir -p $fbank_dir
if [ "$rep_type" = "local" ]; then
	echo -e "\nExtracting mel filterbank features"
	 python codes/prepare/extract_fbank.py save_rep \
	 --utt_id_fn `realpath $utt_id_fn` \
	 --save_dir `realpath $fbank_dir` \
	 --data_split $dataset_split
fi

echo -e "\n\nExtracting ${rep_type} features from ${model_type} ${model_name} model"
echo -e "for sample set $utt_id_fn"

python codes/prepare/extract_rep.py save_rep \
--model_name $model_name \
--ckpt_pth `realpath $ckpt_pth` \
--save_dir $save_dir \
--utt_id_fn `realpath $utt_id_fn` \
--model_type $model_type \
--rep_type $rep_type \
--fbank_dir `realpath $fbank_dir` \
--span $span \
--offset $offset \
--mean_pooling $mean_pooling \
--pckg_dir `realpath $pckg_dir`