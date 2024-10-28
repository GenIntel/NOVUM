# NOVUM

Code for the paper `NOVUM: Neural Object Volumes for Robust Object Classification`

## Note
A few issues can still be found in the code. The branch `dev` is updated more frequently. The final release should happen within November. Thanks for your understanding.


## Installation
### Start
```bash
conda create -n novum python=3.9
conda activate novum
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
conda install jupyter
pip install scikit-image matplotlib imageio plotly opencv-python
conda install -c fvcore -c conda-forge fvcore
pip install black usort flake8 flake8-bugbear flake8-comprehensions
pip install "git+https://github.com/facebookresearch/pytorch3d.git@0.6.2"
```

### Dataset prepration
```bash
bash scripts/1_prepare_data.sh
```

### Training
```bash
bash scripts/2_train.sh
```

### Inference
```bash
bash scripts/3a_eval_cls.sh
```

### Model weights

The model weights can be downloaded a the following [link](https://github.com/GenIntel/NOVUM/releases/download/v1.0.0/classification_saved_model_199.pth)



## Citation

```
@inproceedings{jesslen24novum,
	 author  = {Artur Jesslen and Guofeng Zhang and Angtian Wang and Wufei Ma and Alan Yuille and Adam Kortylewski},
	 title   = {NOVUM: Neural Object Volumes for Robust Object Classification},
	 booktitle = {ECCV},
	 year    = {2024}
 }
 ```
