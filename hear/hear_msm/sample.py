"""MSM-MAE sample wrapper for HEAR 2021

1. Make a copy of this sample with your model name prefixed with MSM-MAE parameters such as `80x208p16x16_mymodel`.
2. Replace `model_path` in `def load_model(model_path="sample/checkpoint-100.pth", mode="all")` with the **bsolute path* of your pre-trained weight.
3. Install your copy as a local module on the hear folder (the parent of this folder): `pip install -e .`
4. Run the `heareval`.
"""

import sys
sys.path.append('../..')
import torch

from .msm_mae.runtime import RuntimeMAE


def load_model(model_path="sample/checkpoint-100.pth", mode="all"):
    model = RuntimeMAE(weight_file=model_path)
    if torch.cuda.is_available():
        model.cuda()
    return model


def get_scene_embeddings(audio, model):
    model.eval()
    with torch.no_grad():
        return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    model.eval()
    with torch.no_grad():
        return model.get_timestamp_embeddings(audio)
