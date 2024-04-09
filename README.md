# Model Composition for Multimodal Large Language Models

## Contents
- [Install](#install)
- [Preparation](#preparation)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

1. Clone this repository and navigate to ModelCompose folder
```bash
git clone https://github.com/THUNLP-MT/ModelCompose.git
cd ModelCompose
```

2. Install Package
```bash
conda create -n modelcompose python=3.10 -y
conda activate modelcompose
pip install -r requirements.txt
```

## Preparation

1. Data

Before training or evaluation, please prepare the datasets based on your need. Json files can be downloaded from [Hugging Face](https://huggingface.co/datasets/Adu2021/ModelCompose/tree/main).

You can organize them under `./data` as follows:
```
data
├── test
│   └── [json files]
├── train
│   └── [json files]
├── evaluation_datasets
├── audiocaps
│   ├── train
|   └── test
├── clotho
│   └── audio
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
├── activitynet
├── PointCloud
│   └── 8192_npy
├── WavCaps
│   └── audios
|       ├── AudioSet_SL_flac
|       ├── BBC_Sound_Effects_flac
|       ├── FreeSound_flac
|       └── SoundBible_flac
└── vg
    ├── VG_100K
    └── VG_100K_2
```

2. Base model

We use vicuna-7b-v1.5 as our base model. [Download](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main)

## Train

We apply a two-stage training paradigm. Find pretrain and finetune scripts under `./scripts/model_composition/train`. Please modify the following parameters in the scripts: `BASE_PATH`, `ROOT`, and `MODEL_BASE`.

Note that we use Video-LLaVA-Pretrain-7B as pretrained checkpoint for text-video modalities. Download pretrained checkpoints for [Video-LLaVA-Pretrain-7B](https://huggingface.co/LanguageBind/Video-LLaVA-Pretrain-7B/tree/main) if needed.

We have released our trained checkpoints at [Hugging Face](https://huggingface.co/Adu2021/ModelCompose/tree/main).

## Evaluation

1. Merge checkpoints

Seperately trained checkpoints should be merged before evaluation. Specify *parameter adjustment coefficient* in `--strategy` param starts with `online-merge-reset-default-`. Use *vision, video, audio, point* for each modality.

```bash
python scripts/model_composition/merge_unimodal_modelcompose.py \
                checkpoints/multimodal-vicuna-7b-v1.5-video-damc \
                checkpoints/multimodal-vicuna-7b-v1.5-audio-damc \
                checkpoints/multimodal-vicuna-7b-v1.5-vision-damc \
                -o checkpoints/multimodal-pdt-video-image-audio \
                --strategy online-merge-reset-default-video=0.333,default-audio=0.333,default-vision=0.333
```

2. Run evaluation

Note that the basename of the checkpoint should contain "multimodal" to load correctly. Replace "multimodal-checkpoint-name" in the following command with your merged checkpoint. 

**AVQA**

```bash
bash scripts/model_composition/test/avqa.sh \
    0,1,2,3,4,5,6,7 \
    multimodal-checkpoint-name \
    [modal] \
    path/to/vicuna-7b-v1.5
    
```

Choose [modal] from `[audio, image, video, image+audio, image+video, video+audio, video+image+audio]`.

Replace "0,1,2,3,4,5,6,7" with actual available gpu.

**MUSIC-AVQA**

```bash
bash scripts/model_composition/test/music_avqa_video+image+audio.sh \
    0,1,2,3,4,5,6,7 \
    multimodal-checkpoint-name \
    path/to/vicuna-7b-v1.5
```

Find scripts for other modalities combinations under `scripts/model_composition/test`.

**MCUB**


```bash
bash scripts/model_composition/test/MCUB-4.sh \
    0,1,2,3,4,5,6,7 \
    multimodal-checkpoint-name \
    path/to/vicuna-7b-v1.5

bash scripts/model_composition/test/MCUB-3.sh \
    0,1,2,3,4,5,6,7 \
    multimodal-checkpoint-name \
    [modal] \
    path/to/vicuna-7b-v1.5
```

Choose [modal] from `[image-audio-video, audio-video-pointcloud, image-audio-pointcloud, image-video-pointcloud]`.

## Citation

If you find our work useful, please consider giving this repository a star and citing our paper.

```
@misc{chen2024model,
      title={Model Composition for Multimodal Large Language Models}, 
      author={Chi Chen and Yiyang Du and Zheng Fang and Ziyue Wang and Fuwen Luo and Peng Li and Ming Yan and Ji Zhang and Fei Huang and Maosong Sun and Yang Liu},
      year={2024},
      eprint={2402.12750},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

[LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon, and it offers strong language & vision abilities.