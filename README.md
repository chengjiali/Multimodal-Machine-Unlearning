
# MultiDelete for Multimodal Machine Unlearning

#### Authors: 
- [Jiali Cheng](https://chengjiali.github.io/) (jiali_cheng@uml.edu)
- [Hadi Amiri](https://cs.uml.edu/~hadi/) (hadi_amiri@uml.edu)

#### MultiDelete Paper: [ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5743_ECCV_2024_paper.php), [Preprint](https://arxiv.org/abs/2311.12047)


## Overview 

We propose *MultiDelete*, the first machine unlearning method that targets unlearning multimodal data and models (MLLM). It formulates multimodal unlearning as 1) Modality Decoupling, 2) Multimodal Knowledge Retention, 3) Unimodal Knowledge Retention.

<p align="center">
    <img src="images/fig1.png" width="1000" align="center">
</p>



## How to run

1. Step 1. Train original model
```bash
bash bash/ori.sh
```

2. Step 2. Unlearn
```bash
python bash/run.py
```


## Citation

If you find *MultiDelete* useful for your research, please consider citing this paper:

```
@inproceedings{cheng2024multidelete,
author="Cheng, Jiali
and Amiri, Hadi",
title="MultiDelete for Multimodal Machine Unlearning",
booktitle="Computer Vision -- ECCV 2024",
year="2025",
publisher="Springer Nature Switzerland",
isbn="978-3-031-72940-9"
}
```
