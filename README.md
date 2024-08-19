<div align="center">
<h3> Pest-ConFormer: A hybrid CNN-Transformer architecture for large-scale
multi-class crop pest recognition</h3>

</div>

This repo is the official implementation of [Pest-ConFormer: A hybrid CNN-Transformer architecture for large-scalemulti-class crop pest recognition]. 

## Pretrain checkpoints
The following table provides pretrained checkpoints in the paper.
| | convmae-Base|
| :---: | :---: |
| pretrained checkpoints| [download](https://drive.google.com/file/d/1AEPivXw0A0b_m5EwEi6fg2pOAoDr8C31/view?usp=sharing) |


### Data preparation

You can download the IP102 [here](https://github.com/xpwu95/IP102) and prepare the IP102 follow this format:

```tree data
IP102
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
```
### Training
To train on IP102, run the following on single node with 2 GPUs:
```bash
python -m torch.distributed.launch --nproc_per_node=2 main_train.py \
    --batch_size 32 \
    --model convvit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 150 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IP102_DIR}
```

### Citation

If you find this paper useful in your research, please consider citing:
```
@ARTICLE{124833,
author= {Mingwei Fang, Zhiping Tan, Yu Tang, Weizhao Chen, Huasheng Huang, Sathian Dananjayan, Yong He, Shaoming Luo},
journal={Expert Systems With Applications},
title={Pest-ConFormer: A hybrid CNN-Transformer architecture for large-scale multi-class crop pest recognition},
year={2024},
doi= {10.1016/j.eswa.2024.124833}
}
```