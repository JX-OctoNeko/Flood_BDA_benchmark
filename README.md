# Flood_BDA_benchmark


## Prerequisites

  opencv-python
  pytorch  
  torchvision  
  pyyaml  
  scikit-image 
  scikit-learn 
  scipy
  tqdm

Tested using Python 3.11 on Ubuntu 22.04.

## Get Started

In `src/constants.py`, change the dataset locations to your own.

### Data Preprocessing

In `data/xView` there are preprocessing scripts for xView datasets

A few configuration files regarding different models are provided in the `configs/` folder for ease of use. *Note that the hyperparameters are not elaborately investigated to obtain a fully optimized performance.*

For semi-supervised there are five hyperparameter could be selected in "config_xview.yaml":
- Strategy1
- Strategy2
- Strategy3
- Strategy4
- None


### Model Training

To train a model from scratch, use

```bash
cd ./src
python train.py train --exp_config PATH_TO_CONFIG_FILE
```

To resume training from some checkpoint, run the code with the `--resume` option.

```bash
python train.py train --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT
```

### Model Evaluation

To evaluate a model on the test subset, use

```bash
python train.py eval --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT --save_on --subset test
```