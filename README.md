![key_visual](misc/key-visual-msm-mae.png)

# Masked Spectrogram Modeling using Masked Autoencoders for Learning General-purpose Audio Representation

This is a demo implementation of Masked Spectrogram Modeling using Masked Autoencoders (MSM-MAE),
a self-supervised learning method for general-purpose audio representation, includes:

- Training code that can pre-train models with arbitrary audio files.
- Evaluation code to test models under two benchmarks, [HEAR 2021](https://arxiv.org/abs/2203.03022) and [EVAR](https://github.com/nttcslab/eval-audio-repr).
- Visualization examples and a notebook.
- Pre-trained weights.

If you find MSM-MAE useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@article{niizumi2022masked,
    title   = {Masked Spectrogram Modeling using Masked Autoencoders for Learning General-purpose Audio Representation}, 
    author  = {Daisuke Niizumi and Daiki Takeuchi and Yasunori Ohishi and Noboru Harada and Kunio Kashino},
    journal = {arXiv:2204.12260},
    year    = {2022},
    eprint  = {2204.12260},
    url     = {https://arxiv.org/abs/2204.12260},
    archivePrefix = {arXiv},
    primaryClass = {eess.AS}
}
```

## 1. Getting Started

The repository relies on the codes from [facebookresearch/mae](https://github.com/facebookresearch/mae), and we patch our changes on these files.

1. Download external source files from [facebookresearch/mae](https://github.com/facebookresearch/mae), and apply a patches.

```sh
curl -o util/lars.py https://raw.githubusercontent.com/facebookresearch/mae/6a2ba402291005b003a70e99f7c87d1a2c376b0d/util/lars.py
curl -o util/lr_decay.py https://raw.githubusercontent.com/facebookresearch/mae/6a2ba402291005b003a70e99f7c87d1a2c376b0d/util/lr_decay.py
curl -o util/lr_sched.py https://raw.githubusercontent.com/facebookresearch/mae/6a2ba402291005b003a70e99f7c87d1a2c376b0d/util/lr_sched.py
curl -o util/misc.py https://raw.githubusercontent.com/facebookresearch/mae/6a2ba402291005b003a70e99f7c87d1a2c376b0d/util/misc.py
curl -o util/pos_embed.py https://raw.githubusercontent.com/facebookresearch/mae/6a2ba402291005b003a70e99f7c87d1a2c376b0d/util/pos_embed.py
curl -o main_pretrain.py https://raw.githubusercontent.com/facebookresearch/mae/6a2ba402291005b003a70e99f7c87d1a2c376b0d/main_pretrain.py
curl -o msm_mae/engine_pretrain.py https://raw.githubusercontent.com/facebookresearch/mae/6a2ba402291005b003a70e99f7c87d1a2c376b0d/engine_pretrain.py
curl -o msm_mae/models_mae.py https://raw.githubusercontent.com/facebookresearch/mae/6a2ba402291005b003a70e99f7c87d1a2c376b0d/models_mae.py
cd util
patch -p1 < patch_util.diff
cd ../msm_mae
patch -p1 < patch_msm_mae.diff
cd ..
patch -p1 < patch_main.diff
```

2. If you need a clean environment, the following [anaconda](https://www.anaconda.com/products/distribution) example creates a new environment named `ar`:

```sh
conda create -n ar python==3.8
conda activate ar
```

3. Install external modules listed on [requirements.txt](requirements.txt).

## 2. Evaluating MSM-MAE

### 2-1. Evaluating on HEAR 2021 NeurIPS Challenge Tasks

We evaluate our models on [our paper](https://arxiv.org/abs/2204.12260) using [hear-eval-kit](https://github.com/neuralaudio/hear-eval-kit) from on [HEAR 2021 NeurIPS Challenge](https://arxiv.org/abs/2203.03022) as follows.

NOTE: The folder `hear` has all the files we need to evaluate models on hear-eval-kit.

1. Install hear-eval-kit as follows:

    ```sh
    pip install heareval
    ```

2. Download your copy of downstream dataset for 16 kHz. See [HEAR NeurIPS 2021 Datasets@zenodo](https://zenodo.org/record/5885750#.YoRfYvPP1zV) for the detail.

3. To evaluate our models, we need a local package which loads and runs our models. The followings shows an example for a model named `80x208p16x16_mymodel`.

    ```sh
    cd hear/hear_msm
    cp sample.py 80x208p16x16_mymodel.py
    ** Edit the 80x208p16x16_mymodel.py here, so that the value of `model_path` points to your model with an absolute path. **
    cd ..
    pip install -e .
    ```

4. We are on the folder `hear`. Run the [hear-eval-kit](https://github.com/neuralaudio/hear-eval-kit) with your pre-trained model.

    ```sh
    python -m heareval.embeddings.runner hear_msm.80x208p16x16_mymodel --tasks-dir /your_copy/hear/tasks
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m heareval.predictions.runner embeddings/hear_msm.80x208p16x16_mymodel/*
    ```

### 2-2. Evaluating on EVAR

[EVAR](https://github.com/nttcslab/eval-audio-repr) is an evaluation package for audio representations used by our research papers such as [BYOL-A](https://arxiv.org/abs/2103.06695) and [Composing General Audio Representation by Fusing Multilayer Features of a Pre-trained Model](https://arxiv.org/abs/2205.08138).
It supports the following downstream tasks: ESC-50, US8K, FSD50K, SPCV1/V2, VoxForge, VoxCeleb1, CREMA-D, GTZAN, NSynth instrument family, Pitch Audio Dataset (Surge synthesizer).

The following steps setup MSM-MAE on the EVAR.

1. Clone EVAR repository and prepare basic items.

    ```sh
    git clone https://github.com/nttcslab/eval-audio-repr.git evar
    cd evar
    curl https://raw.githubusercontent.com/daisukelab/general-learning/master/MLP/torch_mlp_clf2.py -o evar/utils/torch_mlp_clf2.py
    curl https://raw.githubusercontent.com/daisukelab/sound-clf-pytorch/master/for_evar/sampler.py -o evar/sampler.py
    curl https://raw.githubusercontent.com/daisukelab/sound-clf-pytorch/master/for_evar/cnn14_decoupled.py -o evar/cnn14_decoupled.py
    cd ..
    ```

2. Install MSM-MAE files on the cloned `evar` folder.

    ```sh
    ln -s ../../to_evar/ar_msm_mae.py evar/evar
    ln -s ../../to_evar/msm_mae.yaml evar/config
    cd evar
    sed 's/import evar\.ar_byola/import evar\.ar_byola, evar\.ar_msm_mae/' -i lineareval.py
    cd ..
    ```

3. Setup downstream task datasets according to [Preparing-datasets.md](https://github.com/nttcslab/eval-audio-repr/blob/main/Preparing-datasets.md). The following is an example for setting up CREMA-D dataset.

    ```sh
    cd evar
    python evar/utils/download_cremad.py downloads/cremad
    python prepare_wav.py downloads/cremad work/16k/cremad 16000
    cd ..
    ```

Once setup is complete, you can evaluate your models as follows.

- For evaluating a model with an absolute path `/your/path/to/model.pth`.

    ```sh
    cd evar
    python lineareval.py config/msm_mae.yaml cremad weight_file=/your/path/to/model.pth
    ```

- If you want to save GPU memory, set a fewer batch size as follows. This example sets it as 16.

    ```sh
    cd evar
    python lineareval.py config/msm_mae.yaml cremad batch_size=16,weight_file=/your/path/to/model.pth
    ```

## 3. Training From Scratch

First, you need to prepare pre-training data samples; then, you can pre-train your model.
The following is an example to pre-train a model on [FSD50K](https://arxiv.org/abs/2010.00475) dataset.

1. Preprocess .wav files into log-mel spectrogram .npy files. The following converts from a source folder `/your/local/fsd50k/FSD50K.dev_audio` to a new folder `lms_fsd50kdev`.

    ```sh
    python wav_to_lms.py /your/local/fsd50k/FSD50K.dev_audio lms_fsd50kdev
    ```

2. Create a CSV file which is used as a list of pre-training samples containing a single column `file_name`. The following example creates `trainingfiles.csv`.

    ```sh
    echo file_name > trainingfiles.csv
    (cd lms_fsd50kdev && find . -name "*.npy") >> trainingfiles.csv
    ```

3. Make some copies of samples for visualization. The following example makes 3 symbolic links. All the samples under the folder `vis_samples` will be used to visualize reconstruction results of a checkpoint during training.

    ```sh
    mkdir -p lms_fsd50kdev/vis_samples
    ln -s ../1240.npy lms_fsd50kdev/vis_samples/1240.npy
    ln -s ../106237.npy lms_fsd50kdev/vis_samples/106237.npy
    ln -s ../119949.npy lms_fsd50kdev/vis_samples/119949.npy
    ```

4. Once your data is ready, start pre-training as follows.

    ```sh
    python main_pretrain.py 80x208p16x16_your_test --input_size 80x208 --data_path lms_fsd50kdev --dataset trainingfiles --model=mae_vit_base_patch16x16
    ```

### 3-1. About run folders

The training loop creates a folder called the `run folder` to store artifacts of your training run, such as a log, visualizations, and checkpoints.
In the example above, the `80x208p16x16_your_test` is the run folder name which has a format below:

    <input_size>p<patch_size>_<your_free_string>
     - input_size: Two integers concatenated with `x`.
     - patch_size: Two integers concatenated with `x`.
     - your_free_string: Optional.

**NOTE: When a model is evaluated by EVAR or hear-eval-kit, the input and patch size parameters written in the run name are used.**

### 3-2. Evalidation during training

The training loop takes two actions for evaluating checkpoints during training: visualization and calling an external command `quick_eval.sh` (EVAR by default).

- While training, `main_pretrain.py` will call a script named `quick_eval.sh` for every 20 epochs by default. You can edit `quick_eval.sh` for your purpose.

- The training loop also visualizes reconstruction results using checkpoints. You can find them under the sub-folder of the run folder, such as `80x208p16x16_tttt/19`. The folder contains raw results in .npy and image files in .png as the following examples.

    <table><tbody><tr>
    <td><img src="misc/recon_126AbihZt28.png" alt="recon_126AbihZt28" width="250"/></td>
    <td><img src="misc/recon_A1TNpj9hHW0.png" alt="recon_A1TNpj9hHW0" width="250"/></td>
    <td><img src="misc/recon_v1Xn9GONUDE.png" alt="recon_v1Xn9GONUDE" width="250"/></td>
    </tr></tbody></table>


## 4. Visualization Examples

???? An [example notebook](misc/Note_visualization.ipynb) is available. ????

You can try visualizations of reconstruction results as well as attention maps.

- Download `AudioSetWav16k_examples.zip` that contains example wave samples from the [releases](https://github.com/nttcslab/msm-mae/releases) and unzip the zip file in the `misc` folder beforehand.

A reconstruction examples:

![recon512](misc/recon_example1.png)

An attention map examples:

![attn512](misc/attn_example3.png)

## 5. Pre-trainede Weights and Network Structure Details

Three pre-trained weights are published on the [releases](https://github.com/nttcslab/msm-mae/releases),
and the followings are their EVAR task results:

| weight             | esc50   | us8k   | spcv2   | vc1   | voxforge   | cremad   | gtzan   | nsynth   | surge   |
|:-------------------|:--------|:-------|:--------|:------|:-----------|:---------|:--------|:---------|:--------|
| 80x512p16x16_paper | 89.1%   | 84.3%  | 93.1%   | 56.1% | 97.5%      | 70.1%    | 79.3%   | 76.9%    | 37.5%   |
| 80x512p16x16_0425  | 88.9%   | 84.0%  | 92.4%   | 56.5% | 96.8%      | 67.6%    | 81.4%   | 76.1%    | 37.4%   |
| 80x208p16x16_0422  | 87.1%   | 82.2%  | 91.1%   | 49.7% | 95.6%      | 66.9%    | 75.9%   | 76.5%    | 37.8%   |

- 80x512p16x16_paper: This weight is the pre-trained weight used to calculate the numbers in the paper, trained on our old unpolished code. **Cannot be used for visualizations** due to a minor difference in the decoder structure.
- 80x512p16x16_0425: This weight is a pre-trained weight that is trained with the current code to check reproducibility.
- 80x208p16x16_0422: This weight is a pre-trained weight that is trained with the current code to check reproducibility.

**FYI:** You can check [a notebook](misc/Note_calc_evar_score.ipynb) that summarizes EVAR task results.

### 5-1. Network Structure Details

Our ViT network implementation has three minor differences from the original MAE implementation, due to historical reason. (We have already been testing based on [pengzhiliang/MAE-pytorch](https://github.com/pengzhiliang/MAE-pytorch) in late 2021, before the MAE implementation became available.)

- Our implementation follows [BEiT](https://github.com/microsoft/unilm/tree/master/beit), where the k bias is always zero in the Attention calculation. Though [it is explained to be equivalent in terms of calculation results](https://github.com/microsoft/unilm/issues/510), we keep this implementation in our network for better reproducibility.
- `[CLS]` token is removable by the option `--no_cls_token`.
- The decoder uses 1d positional embedding. You can switch to the 2d version by the option `--dec_pos_2d`.


## 6. License

See [LICENSE](LICENSE) for details.

## Acknowledgements

- Our code is based on the [MAE PyTorch/GPU re-implementation](https://github.com/facebookresearch/mae) of the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377).
- We also refer to another implementation of MAE, [pengzhiliang/MAE-pytorch](https://github.com/pengzhiliang/MAE-pytorch).
- We use [nnAudio](https://arxiv.org/abs/1912.12055) ([KinWaiCheuk/nnAudio](https://github.com/KinWaiCheuk/nnAudio)) for converting raw audio into log-mel spectrogram.

We appreciate these publicly available implementations and all the modules our experiments heavily depend on!

## References

- MAE: *[Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll??r, and Ross Girshick "Masked Autoencoders Are Scalable Vision Learners," 2021](https://arxiv.org/abs/2111.06377).*
- MSM-MAE: *[Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, and Kunio Kashino "Masked Spectrogram Modeling using Masked Autoencoders for Learning General-purpose Audio Representation," 2022](https://arxiv.org/abs/2204.12260).*
- BEiT: *[Hangbo Bao, Li Dong, and Furu Wei "BEiT: BERT Pre-Training of Image Transformers," 2021](https://arxiv.org/abs/2106.08254).*
- HEAR 2021: *[Joseph Turian, Jordie Shier, Humair Raj Khan, Bhiksha Raj, Bj??rn W. Schuller, Christian J. Steinmetz, Colin Malloy, George Tzanetakis, Gissel Velarde, Kirk McNally, Max Henry, Nicolas Pinto, Camille Noufi, Christian Clough, Dorien Herremans, Eduardo Fonseca, Jesse Engel, Justin Salamon, Philippe Esling, Pranay Manocha, Shinji Watanabe, Zeyu Jin, and Yonatan Bisk "HEAR 2021: Holistic Evaluation of Audio Representations," 2022](https://arxiv.org/abs/2203.03022).*
- FSD50K: *[Eduardo Fonseca and Xavier Favory and Jordi Pons and Frederic Font and Xavier Serra, ???FSD50K: an Open Dataset of Human-Labeled Sound Events,??? 2020](https://arxiv.org/abs/2010.00475).*
