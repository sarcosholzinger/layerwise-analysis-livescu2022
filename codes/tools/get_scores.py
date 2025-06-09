"""
Get CCA and MI scores
"""

import fire
from glob import glob
import numpy as np
from operator import itemgetter
import os
import random
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr
import time
from tqdm import tqdm

import scripts.codes.tools.tools as tools

curr_dir = os.path.dirname(os.path.realpath(__file__))
import sys

sys.path.insert(0, os.path.join(curr_dir, ".."))
from scripts.codes.utils import save_dct, read_lst, format_time, load_dct, add_to_file
from scripts.codes.tools.tools_utils import LAYER_CNT


class getCCA:
    def __init__(
        self,
        model_name,
        fbank_dir,
        rep_dir,
        exp_name,
        base_layer=0,
        rep_dir2=None,
        embed_dir=None,
        sample_data_fn=None,
        span="phone",
        mean_score=False,
        eval_single_layer=False,
        layer_num=-1,
        instance_cap=None, # cap on number of instances per word
    ):
        """
        exp_name: cca-mel | cca-intra | cca-inter | cca-glove | cca-agwe
        """
        print(eval_single_layer)
        if eval_single_layer:
            assert layer_num != -1
        self.layer_num = layer_num
        self.eval_single_layer = eval_single_layer
        self.num_conv_layers = LAYER_CNT[model_name]["local"]
        self.num_transformer_layers = LAYER_CNT[model_name]["contextualized"]
        self.fbank_dir = fbank_dir
        self.rep_dir = rep_dir
        self.base_layer = base_layer
        self.rep_dir2 = rep_dir2
        self.embed_fn = os.path.join(embed_dir, f'{exp_name.split("_")[-1]}_embed.pkl')
        self.sample_data_fn = sample_data_fn
        self.model_name = model_name
        self.score_dct = {}
        if exp_name in ["cca_glove", "cca_agwe", "cca_word"]:
            assert span == "word"
        elif exp_name == "cca_phone":
            assert span == "phone"
        self.span = span
        self.exp_name = exp_name
        self.mean_score = mean_score
        self.instance_cap = instance_cap

    def get_score_flag(self, layer_id):
        get_score = False
        if self.eval_single_layer:
            if self.layer_num == layer_id:
                get_score = True
        else:
            get_score = True
        return get_score

    def get_cca_score(
        self,
        view1,
        view2,
        rep_dir,
        layer_id,
        label_lst=None,
        force_train=False,
        subset=None,
    ):
        start_time = time.time()
        sim_score = tools.get_cca_score(
            view1,
            view2,
            rep_dir,
            layer_id,
            self.exp_name,
            label_lst=label_lst,
            subset=subset,
            force_train=force_train,
            mean_score=self.mean_score,
        )
        self.score_dct[layer_id] = sim_score

        print_score = np.round(sim_score, 2)
        if isinstance(layer_id, int):
            layer_type = "Transformer"
            layer_num = layer_id
        elif "C" in layer_id:
            layer_type = "Conv"
            layer_num = layer_id[1:]
        elif "T" in layer_id:
            layer_type = "Transformer"
            layer_num = layer_id[1:]
        print(
            f"[{format_time(start_time)}] {layer_type} layer {layer_num}: {print_score}"
        )
        return sim_score

    def cca_mel(self):
        rep_dir_contextualized = os.path.join(
            self.rep_dir, "contextualized", "frame_level"
        )
        rep_dir_local = os.path.join(self.rep_dir, "local", "frame_level")
        all_fbank = np.load(os.path.join(self.fbank_dir, "all_features.npy"))

        if "avhubert" in self.model_name:
            all_fbank_downsampled = np.load(
                os.path.join(self.fbank_dir, "all_features_downsampled_by4.npy")
            )
        else:
            all_fbank_downsampled = np.load(
                os.path.join(self.fbank_dir, "all_features_downsampled.npy")
            )
        layer_start = 1

        for layer_id in range(1, self.num_conv_layers + 1):
            if self.get_score_flag(f"C{layer_id}"):
                start_time = time.time()
                fname = "layer_" + str(layer_id) + ".npy"
                rep_mat = np.load(os.path.join(rep_dir_local, fname))
                if layer_id != self.num_conv_layers:  # downsample model representations
                    view1 = all_fbank.T
                    subset = "downsampled"
                else:
                    view1 = all_fbank_downsampled.T
                    subset = "original"
                sim_score = self.get_cca_score(
                    view1,
                    rep_mat.T,
                    rep_dir_local,
                    f"C{layer_id}",
                    subset=subset,
                )

        for layer_id in range(layer_start, self.num_transformer_layers + 1):
            if self.get_score_flag(f"T{layer_id}"):
                start_time = time.time()
                fname = "layer_" + str(layer_id) + ".npy"
                rep_mat = np.load(os.path.join(rep_dir_contextualized, fname))
                sim_score = self.get_cca_score(
                    all_fbank_downsampled.T,
                    rep_mat.T,
                    rep_dir_contextualized,
                    f"T{layer_id}",
                )

    def cca_intra(self):
        rep_dir = os.path.join(self.rep_dir, "contextualized", "frame_level")
        z_mat = np.load(os.path.join(rep_dir, f"layer_{self.base_layer}.npy"))
        for layer_id in range(1, self.num_transformer_layers + 1):
            if self.get_score_flag(layer_id):
                start_time = time.time()
                c_mat = np.load(os.path.join(rep_dir, f"layer_{layer_id}.npy"))
                sim_score = self.get_cca_score(
                    z_mat.T,
                    c_mat.T,
                    rep_dir,
                    layer_id,
                )

    def cca_inter(self):
        rep_dir1 = os.path.join(self.rep_dir, "contextualized", "frame_level")
        rep_dir2 = os.path.join(self.rep_dir2, "contextualized", "frame_level")
        for layer_id in range(1, self.num_transformer_layers + 1):
            if self.get_score_flag(layer_id):
                start_time = time.time()
                c_mat1 = np.load(os.path.join(rep_dir1, f"layer_{layer_id}.npy"))
                c_mat2 = np.load(os.path.join(rep_dir2, f"layer_{layer_id}.npy"))
                sim_score = self.get_cca_score(
                    c_mat1.T,
                    c_mat2.T,
                    rep_dir1,
                    layer_id,
                    rep_dir2=rep_dir2,
                )

    def get_num_splits(self):
        search_str = self.sample_data_fn.replace("_0.json", "_*.json")
        num_splits = len(glob(search_str))
        assert num_splits != 0, "data not found"
        return num_splits

    def update_label_lst(self, split_num, all_labels, dir_name=None):
        assert dir_name is not None
        fname = os.path.join(dir_name, f"labels_{split_num}.lst")
        label_lst = read_lst(fname)
        all_labels.extend(label_lst)

    def filter_label_lst(self, all_labels, embed_dct):
        label_idx_dct = {}
        num_labels = len(all_labels)
        valid_indices = list(np.arange(num_labels))
        # valid_label_lst = []
        for idx, label in enumerate(all_labels):
            if label not in embed_dct:
                valid_indices.remove(idx)
        for idx in valid_indices:
            label = all_labels[idx]
            _ = label_idx_dct.setdefault(label, [])
            label_idx_dct[label].append(idx)
        print(
            f"{num_labels-len(valid_indices)} of {num_labels} {self.span} segments dropped"
        )
        return valid_indices, label_idx_dct
    
    def update_idx_lst(self, valid_indices, label_idx_dct):
        """
        Update the indices to have less than instance_cap instances
        """
        print("Generating new list of valid indices")
        new_valid_indices = []
        for label in tqdm(list(label_idx_dct.keys())):
            idx_lst = label_idx_dct[label]
            if len(idx_lst) > self.instance_cap:
                label_idx_dct[label] = random.sample(idx_lst, self.instance_cap)
            new_valid_indices.extend(label_idx_dct[label])
        assert not len(new_valid_indices) > len(valid_indices)
        return new_valid_indices

    def cca_embed(self):
        check_cond = 'semantic' in self.exp_name or 'syntactic' in self.exp_name
        if check_cond:
            rep_dir = self.rep_dir
            all_labels = read_lst(os.path.join(rep_dir, "labels.lst"))
            valid_indices, label_idx_dct = self.filter_label_lst(all_labels, embed_dct)
            if self.instance_cap is not None:
                valid_indices = self.update_idx_lst(valid_indices, label_idx_dct)
                valid_label_lst = [all_labels[idx1] for idx1 in valid_indices]
                all_embed = np.array(
                        [embed_dct[all_labels[idx1]] for idx1 in valid_indices]
                    )
        else:
            rep_dir = os.path.join(self.rep_dir, "contextualized", f"{self.span}_level")
            all_labels = []
        embed_dct = load_dct(self.embed_fn)
        num_splits = self.get_num_splits()
        for layer_id in range(self.num_transformer_layers + 1):
            if self.get_score_flag(layer_id):
                if check_cond:
                    all_rep = np.load(os.path.join(rep_dir, f"layer_{self.layer_num}.npy"))
                    all_rep = all_rep[np.array(valid_indices)]
                else:
                    all_rep = []
                    for split_num in range(num_splits):
                        rep_fn = os.path.join(rep_dir, str(split_num), f"layer_{layer_id}.npy")
                        rep_mat = np.load(rep_fn)
                        all_rep.extend(rep_mat)
                        if layer_id == 0 or self.eval_single_layer:
                            self.update_label_lst(split_num, all_labels, rep_dir)

                    all_rep = np.array(all_rep)  # N x d
                    if layer_id == 0 or self.eval_single_layer:
                        valid_indices, _ = self.filter_label_lst(all_labels, embed_dct)
                        all_embed = np.array(
                            [embed_dct[all_labels[idx1]] for idx1 in valid_indices]
                        )
                        valid_label_lst = [all_labels[idx1] for idx1 in valid_indices]
                    all_rep = all_rep[np.array(valid_indices)]
                sim_score = self.get_cca_score(
                    all_rep.T,
                    all_embed.T,
                    rep_dir,
                    layer_id,
                    label_lst=valid_label_lst,
                )

    def cca_word(self):
        self.cca_embed()

    def cca_phone(self):
        self.cca_embed()

    def cca_glove(self):
        self.cca_embed()

    def cca_agwe(self):
        self.cca_embed()
    
    def cca_semantic(self):
        self.cca_embed()
    
    def cca_syntactic(self):
        self.cca_embed()


class getMI:
    def __init__(
        self,
        eval_dataset_split,
        sample_data_dir,
        rep_dir,
        save_fn,
        layer_id,
        span,
        iter_num,
        data_sample,
        num_clusters,
        train_dataset_split=None,
    ):
        self.sample_data_dir = sample_data_dir
        self.rep_dir = rep_dir
        self.save_fn = save_fn
        self.layer_id = layer_id
        self.data_sample = data_sample
        self.iter_num = iter_num

        if "train" in eval_dataset_split:
            self.all_rep, self.all_labels = self.read_data(
                eval_dataset_split
            )  # load train data
            self.eval_rep, self.eval_labels = None, None
        elif "dev" in eval_dataset_split:
            self.all_rep, self.all_labels = self.read_data(
                train_dataset_split
            )  # load train data
            self.eval_rep, self.eval_labels = self.read_data(eval_dataset_split)

        max_iter = 500
        if span == "phone":
            # n_clusters = 500
            n_clusters = num_clusters
            batch_size = 1500
        elif span == "word":
            # n_clusters = 5000
            n_clusters = num_clusters
            batch_size = 4000
        self.mi_score = tools.get_mi_score(
            n_clusters,
            batch_size,
            max_iter,
            eval_dataset_split,
            self.all_rep,
            self.all_labels,
            self.eval_rep,
            self.eval_labels,
        )

    def write_to_file(self, mi_score):
        """
        Saving scores to a file
        """
        with open(self.save_fn, "a") as f:
            f.write(
                ",".join(
                    list(
                        map(
                            str,
                            [
                                self.layer_id,
                                self.data_sample,
                                self.iter_num,
                                np.round(mi_score, 3),
                            ],
                        )
                    )
                )
                + "\n"
            )

    def read_data(self, split):
        rep_dir = self.rep_dir.replace("dev-clean", split)
        sample_data_fn = os.path.join(
            self.sample_data_dir, f"{split}_segments_sample{self.data_sample}_0.json"
        )
        search_str = sample_data_fn.replace("_0.json", "_*.json")
        num_splits = len(glob(search_str))
        assert num_splits != 0
        all_rep, all_labels = [], []
        for idx in range(num_splits):
            rep_fn = os.path.join(rep_dir, str(idx), f"layer_{self.layer_id}.npy")
            rep_mat = np.load(rep_fn)
            all_rep.extend(rep_mat)
            label_lst = read_lst(os.path.join(rep_dir, f"labels_{idx}.lst"))
            all_labels.extend(label_lst)

        all_rep = np.array(all_rep)
        assert len(all_rep) == len(all_labels)
        return all_rep, all_labels


def evaluate_mi(
    eval_dataset_split,
    sample_data_dir,
    rep_dir,
    save_fn,
    layer_id,
    span,
    iter_num,
    data_sample,
    num_clusters,
    train_dataset_split=None,
):
    mi_obj = getMI(
        eval_dataset_split,
        sample_data_dir,
        rep_dir,
        save_fn,
        layer_id,
        span,
        iter_num,
        data_sample,
        num_clusters,
        train_dataset_split,
    )
    mi_obj.write_to_file(mi_obj.mi_score)


def evaluate_cca(
    model_name,
    save_fn,
    fbank_dir,
    rep_dir,
    exp_name,
    base_layer=0,
    rep_dir2=None,
    embed_dir=None,
    sample_data_fn=None,
    span="phone",
    mean_score=False,
    eval_single_layer=False,
    layer_num=-1,
):
    cca_obj = getCCA(
        model_name,
        fbank_dir,
        rep_dir,
        exp_name,
        base_layer,
        rep_dir2,
        embed_dir,
        sample_data_fn,
        span,
        mean_score,
        eval_single_layer,
        layer_num
    )
    getattr(cca_obj, exp_name)()

    if mean_score:
        save_fn = save_fn.replace(".json", "_mean.json")
    
    if eval_single_layer:
        assert len(cca_obj.score_dct) == 1
        sample_num = save_fn.split("_")[-1].split(".")[0][-1]
        save_fn = "_".join(save_fn.split("_")[:-1]) + ".lst"
        add_to_file(
            ",".join(
                list(map(str, [layer_num, sample_num, cca_obj.score_dct[layer_num]]))
            )
            + "\n",
            save_fn,
        )
    else:
        save_dct(save_fn, cca_obj.score_dct)
    print(f"Result saved at {save_fn}")

def evaluate_wordsim(model_name, wordsim_task_fn, embedding_dir, save_fn):
    wordsim_tasks = load_dct(wordsim_task_fn)
    num_transformer_layers = LAYER_CNT[model_name]["contextualized"]
    res_dct = {}
    _ = res_dct.setdefault("micro average", {})
    _ = res_dct.setdefault("macro average", {})
    mean_score = 0
    for layer_num in range(num_transformer_layers + 1):
        embed_dct = load_dct(os.path.join(embedding_dir, f"layer{layer_num}.json"))
        res_dct["micro average"][layer_num] = 0
        res_dct["macro average"][layer_num] = 0
        num_pairs = 0
        for task_name, task_lst in wordsim_tasks.items():
            srho_score = tools.get_similarity_score(task_lst, embed_dct)
            res_dct["micro average"][layer_num] += srho_score * len(task_lst)
            res_dct["macro average"][layer_num] += srho_score
            num_pairs += len(task_lst)
            _ = res_dct.setdefault(task_name, {})
            res_dct[task_name][layer_num] = srho_score
        res_dct["micro average"][layer_num] /= num_pairs
        res_dct["macro average"][layer_num] /= len(wordsim_tasks)
    save_dct(save_fn, res_dct)

def evaluate_spokensts(
        model_name,
        rep_dir,
        gt_score_fn,
        pair_idx_fn,
        res_dir,
        sample_data_fn,
        mean=True, # replicates eval setup of Merkx et al. 
):
    rep_dir = os.path.join(rep_dir, model_name, 'utt_level')
    num_splits = len(glob(os.path.join(sample_data_fn, "split*.tsv")))
    gt_dct = load_dct(gt_score_fn)
    pair_idx_dct = load_dct(pair_idx_fn)
    layer_lst = [item.split('.')[0].split('_')[1] for item in os.listdir(os.path.join(rep_dir, "0"))]
    res_dct = {}
    for layer_num in layer_lst:
        rep_mat = []
        for split_idx in range(num_splits):
            rep_mat.extend(np.load(os.path.join(rep_dir, str(split_idx), f"layer_{layer_num}.npy")))
        rep_mat = np.array(rep_mat)
        cosine_similarities, human_judgements = [], []
        for pair_name in gt_dct:
            all_cos_sim = []
            for idx1, idx2 in pair_idx_dct[pair_name]:
                all_cos_sim.append(1 - cosine(rep_mat[idx1], rep_mat[idx2]))
            assert len(all_cos_sim) == 16
            if mean:
                cosine_similarities.append(np.mean(all_cos_sim))
                human_judgements.append(gt_dct[pair_name])
            else:
                cosine_similarities.extend(all_cos_sim)
                human_judgements.extend([gt_dct[pair_name]]*len(all_cos_sim))
    
        srho_score, _ = spearmanr(np.array(cosine_similarities), np.array(human_judgements))
        res_dct[layer_num] = srho_score
    save_dct(os.path.join(res_dir, f'{model_name}_spoken_sts.json'), res_dct)

if __name__ == "__main__":
    fire.Fire(
        {
            "mi": evaluate_mi,
            "cca": evaluate_cca,
            "wordsim": evaluate_wordsim,
            "spoken_sts": evaluate_spokensts,
        }
    )
