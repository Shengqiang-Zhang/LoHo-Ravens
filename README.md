# LoHoRavens

This repository is based on the minibatch training enabled CLIPort
repository [CLIPort-Batchify](https://github.com/ChenWu98/cliport-batchify).
The video and homepage of this project are available at [this website](https://cisnlp.github.io/lohoravens-webpage/).

Warnings: I'm currently refactoring the code and write documentation for this repo. Feel free to provide feedback to me.
Thanks for your patience to explore this code repo.


## Installation (overriding the original repo)

1. Create environment:

```bash
conda create -n lohoravens python=3.8
```

2. Activate environment:

```bash
conda activate lohoravens
```

3. Clone LoHoRavens

```bash
git clone https://github.com/Shengqiang-Zhang/lohoravens.git
```

4. Install from requirements:

```bash
conda install pip
cd LoHo-Ravens
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pytorch-lightning==1.9.5
export CLIPORT_ROOT=$(pwd)
python setup.py develop
```

TODO: add config dependencies on LLM and VLM.

<br>

## Generate dataset

```bash
python cliport/demos.py n=1000 \
    task=pick-and-place-primitive \
    mode=train
```

In addition, we provide the method to generate the step-by-step instructions and corresponding images,
which could be used for fine-tuning LLaVA.
To do this, just set `Task.generate_instruction_for_each_step=True` and the save paths for instructions and images
in the file `cliport/tasks/task.py`.

## Train the primitive

```bash
python cliport/train.py train.task=lohoravens-pick-and-place-primitive \
    train.agent=cliport \
    train.attn_stream_fusion_type=add \
    train.trans_stream_fusion_type=conv \
    train.lang_fusion_type=mult \
    train.lr="${TRAIN_LR}" \
    train.batch_size="${BATCH_SIZE}" \
    train.gpu="${GPU_SIZE}" \
    train.n_demos=1000 \
    train.n_steps=601000 \
    train.n_val=200 \
    dataset.cache=False \
    train.exp_folder=${EXP_FOLDER} \
    train.data_dir=${ONE_K_DATA_FOLDER} \
    dataset.type=multi \
    train.log=True  \
    
```

## Evaluate the primitive

```bash
python cliport/eval.py model_task=lohoravens-pick-and-place-primitive \
    eval_task=lohoravens-pick-and-place-primitive \
    agent=cliport \
    mode=${mode} \
    n_demos=200 \
    train_demos=1000 \
    checkpoint_type=${CKPT} \
    type=multi \
    model_dir=${MODEL_DIR} \
    exp_folder=exps \
    data_dir=${DATA_DIR} \

```

## Use LLM and VLM

```bash
python eval_llm_vlm.py
```

## Acknowledgements

This work use code from the following open-source projects and datasets:

#### CLIPort-batchify

Original: [https://github.com/ChenWu98/cliport-batchify](https://github.com/ChenWu98/cliport-batchify)

#### Google Ravens (TransporterNets)

Original:  [https://github.com/google-research/ravens](https://github.com/google-research/ravens)  
License: [Apache 2.0](https://github.com/google-research/ravens/blob/master/LICENSE)    
Changes: All PyBullet tasks are directly adapted from the Ravens codebase. The original TransporterNets models were
reimplemented in PyTorch.

#### OpenAI CLIP

Original: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)  
License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)  
Changes: Minor modifications to CLIP-ResNet50 to save intermediate features for skip connections.

#### Google Scanned Objects

Original: [Dataset](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects)  
License: [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
Changes: Fixed center-of-mass (COM) to be geometric-center for selected objects.

#### U-Net

Original: [https://github.com/milesial/Pytorch-UNet/](https://github.com/milesial/Pytorch-UNet/)  
License: [GPL 3.0](https://github.com/milesial/Pytorch-UNet/)  
Changes: Used as is in [unet.py](cliport/models/core/unet.py). Note: This part of the code is GPL 3.0.

## Citations

**LoHoRavens**

```bibtex
@article{zhang2023lohoravens,
  title={LoHoRavens: A Long-Horizon Language-Conditioned Benchmark for Robotic Tabletop Manipulation},
  author={Zhang, Shengqiang and Wicke, Philipp and {\c{S}}enel, L{\"u}tfi Kerem and Figueredo, Luis and Naceri, Abdeldjallil and Haddadin, Sami and Plank, Barbara and Sch{\"u}tze, Hinrich},
  journal={arXiv preprint arXiv:2310.12020},
  year={2023}
}
```

**CLIPort**

```bibtex
@inproceedings{shridhar2021cliport,
  title     = {CLIPort: What and Where Pathways for Robotic Manipulation},
  author    = {Shridhar, Mohit and Manuelli, Lucas and Fox, Dieter},
  booktitle = {Proceedings of the 5th Conference on Robot Learning (CoRL)},
  year      = {2021},
}
```

**CLIP**

```bibtex
@article{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  journal={arXiv preprint arXiv:2103.00020},
  year={2021}
}
```

**TransporterNets**

```bibtex
@inproceedings{zeng2020transporter,
  title={Transporter networks: Rearranging the visual world for robotic manipulation},
  author={Zeng, Andy and Florence, Pete and Tompson, Jonathan and Welker, Stefan and Chien, Jonathan and Attarian, Maria and Armstrong, Travis and Krasin, Ivan and Duong, Dan and Sindhwani, Vikas and others},
  booktitle={Proceedings of the 4th Conference on Robot Learning (CoRL)},
  year= {2020},
}
```

## Issues?

For questions and issues beyond the [original repository](https://github.com/cliport/cliport) and
the [CLIPort-batchify repository](https://github.com/ChenWu98/cliport-batchify), you can create
issues or email [me](https://github.com/Shengqiang-Zhang).
