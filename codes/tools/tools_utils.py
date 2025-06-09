LAYER_CNT = {
    "wav2vec_small": {"contextualized": 12, "local": 7},
    "wav2vec_vox": {"contextualized": 24, "local": 7},
    "hubert_small": {"contextualized": 12, "local": 7},
    "hubert_large": {"contextualized": 24, "local": 7},
    "hubert": {"contextualized": 12, "local": 0}, #added for layerwise analysis (SAH)
    "avhubert_small_lrs3": {"contextualized": 12, "local": 0},
    "avhubert_small_lrs3_vc2": {"contextualized": 12, "local": 0},
    "avhubert_large_lrs3_vc2": {"contextualized": 24, "local": 0},
    "wavlm_small": {"contextualized": 12, "local": 7},
    "wavlm_large": {"contextualized": 24, "local": 7},
    "fastvgs_coco": {"contextualized": 8, "local": 0},
    "fastvgs_places": {"contextualized": 8, "local": 0},
    "fastvgs_plus_coco": {"contextualized": 12, "local": 0},
    "xlsr53_56": {"contextualized": 24, "local": 7},
    "xlsr128_300m": {"contextualized": 24, "local": 7},
}
