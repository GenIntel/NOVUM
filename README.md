# NOVUM

Code for the paper `NOVUM: Neural Object Volumes for Robust Object Classification`



## Installation
### Start
```bash
conda create -n xnovum python=3.12
conda activate xnovum
conda install pytorch-gpu==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c iopath iopath
conda install pytorch3d -c pytorch3d
conda install jupyter
pip install scikit-image matplotlib imageio plotly opencv-python
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

## Acknowledgements

Special thanks to Nhi Pham for the help with the codebase. 
```