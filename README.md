<p align="center">
  <h1 align="center">SEAL 🦭<br>
Segment Any Events with Language</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/seungjun-lee-43101a261/">Seungjun Lee</a></span> ·  
    <a href="https://www.comp.nus.edu.sg/~leegh/">Gim Hee Lee</a><sup></sup> <br>
    Department of Computer Science, National University of Singapore<br>
  </p>
  <h2 align="center">ICLR 2026</h2>
  <h3 align="center"><a href="https://github.com/0nandon/SEAL">Code</a> | <a href="https://arxiv.org/abs/2601.23159">Paper</a> | <a href="https://0nandon.github.io/SEAL/">Project Page</a> </h3>
  <div align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
  </div>
</p>

<p align="center">
  <a href="">
    <img src="https://github.com/0nandon/SEAL/blob/main/static/teaser.png" alt="Logo" width="100%">
  </a>
</p>
<p align="center">
Our <strong>SEAL</strong> is the first Semantic-aware Segment Any Events model. 
</p>
</p>

<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#todo">TODO</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li>
      <a href="#training">Training</a>
    </li>
     <li>
      <a href="#evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#demo">Demo</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>

## News:

- [2026/01/26] SEAL is accepted to ICLR 2026 🔥. The code will be released before April.
- [2026/03/31] Code for SEAL and instance segmentation benchmarks are released 👊🏻! Interactive demo is coming soon.

## TODO
- [x] Release the code of SEAL
- [x] Release the benchmarks for instance segmentation
- [ ] Release the interactive demo.
- [ ] Release the code of SEAL++
- [ ] Release the benchmarks for part segmentation

## Installation
### Dependencies :memo:
The main dependencies of the project are the following:
```yaml
python: 3.8
cuda: 11.8
```
You can set up a conda environment as follows:
```
conda create -n seal python=3.8
conda activate seal

pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

pip install git+https://github.com/openai/CLIP.git

cd pointnet2
pip install . --force-reinstall --no-deps
cd ..

# For demo
pip install gradio
```

## Data Preparation

The training datasets, benchmarks and pretrained weights are available <a href="https://huggingface.co/datasets/onandon/SEAL/tree/main">here</a>. You can easily download all the preprocessed data by running:
```
python download_data.py
python download_checkpoints.py
```

Once you run the above command, the downloaded files should be located to designated path. Refer to the file structure below:
```
cache
├── eventsam.pth                        <- pretrained weights of events backbone
├── seal.pt                             <- pretrained weights of seal
└── sam_vit_b_01ec64.pth                        

...

dataset
├── data                   
│   ├── ddd17                           <- DDD17 dataset
│   │   ├── dir0                        <- Driving scene
│   │   │   ├── imgs                    <- RGB images
│   │   │   ├── mhsg                    <- MHSG guidance
│   │   │   ├── recons_image            <- E2VID-reconstructed frames
│   │   │   ├── segmentation_masks      <- Semantic maps
│   │   │   └── voxel_image             <- Events frame
│   │   ├── dir1
│   │   └── ...
│   └── DSEC                            <- DSEC dataset
│       ├──test                         <- Test scene
│       │  ├── zurich_city_13_a         
│       │  │   ├── instance_gt          <- Ground-truth instance masks
│       │  │   ├── recons_image         <- E2VID-reconstructed frames
│       │  │   ├── rgb_image            <- RGB images
│       │  │   ├── semantic_gt          <- Semantic maps
│       │  │   └── voxel_image          <- Events frame
│       │  └── ...   
│       └── train
│           └── ...
└── metadata
    ├──ddd17_eval.txt                   <- Evaluation lists of DDD17-Instance
    ├──dsec11_eval.txt                  <- Evaluation lists of DSEC11-Instance
    ├──dsec19_eval.txt                  <- Evaluation lists of DSEC19-Instance
    └──train.txt                        <- Training list


```

## Training

Training consists of two stages. We conduct the training in RTX A6000 Ada GPU.

#### 1. Training events backbone

You need to train events backbone by following the EventSAM. Enter the path to store the checkpoints in <a href="https://github.com/0nandon/SEAL/blob/064a50320dbb858769a4bf3101a940bc764c6bd9/event_encoder/train_eventsam.py#L94">here<a> of event_encoder/train_eventsam.py.
```
python event_encoder/train_eventsam.py
```

Or, you may directly use the pretrained weights we provide in `cache/eventsam.pth`. In that case, skip Stage 1 and run Stage 2 directly.

#### 2. Training SEAL
```
python event_encoder/train_seal.py --config-path=../configs/train --config-name=seal.yaml
```
The model weight and log will be stored in `checkpoints` and `log` folder, respectively. We also provide our training log in `train.log`.

## Evaluation

We provide pretrained weights at `cache/seal.pt`. By running the commands below, the code will evaluate SEAL using the provided weights. To use your own weights, simply append `general.checkpoint={YOUR_PATH}` to the end of the command.

In the commands below, specify the dataset name: `DDD17`, `DSEC11`, or `DSEC19` in `{DATASET}`.

#### Evaluate with box prompt

```
python event_encoder/test_seal.py --config-path=../configs/eval/{DATASET} --config-name=seal.yaml
```

#### Evaluate with point prompt
```
python event_encoder/test_seal.py --config-path=../configs/eval/{DATASET} --config-name=seal.yaml dataset.test.prompt=point
```

## Demo

Demo is coming soon!
```
python gradio_seal.py --config configs/eval/DSEC19/seal.yaml --checkpoint cache/seal.pt --data-root dataset_/data/DSEC/test --server-port 7860
```

## Acknowledgement

Our work is inspired a lot from the following works. We sincerely appreciate to their great contributions!

* <a href="https://github.com/zhiwen-xdu/EventSAM">EventSAM</a>
* <a href="https://github.com/ldkong1205/OpenESS">OpenESS</a>
* <a href="https://github.com/Hatins/DEOE">DEOE</a>
* <a href="https://github.com/uzh-rpg/rvt">RVT</a>

## Citation
If you find our code or paper useful, please cite
```bibtex
@article{lee2026segment,
  title={Segment Any Events with Language},
  author={Lee, Seungjun and Lee, Gim Hee},
  journal={arXiv preprint arXiv:2601.23159},
  year={2026}
}
```
