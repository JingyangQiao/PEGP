## Usage

First, clone our repository:

```
git clone https://github.com/JingyangQiao/PEGP
cd PEGP/clip-pegp-prompt
```

Then, install the packages with ```environment.yaml``` by: 

```
conda env create -f environment.yaml
```

Notice: We produce our results on NVIDIA RTX 4090 GPU, and the CUDA version is 12.2. 

## Data preparation

Pass your dataset path to  `data_path` in ```configs/cifar100_split.json``` file or ```configs/imagenet100_split.json```.

## Training

To train a model via command line:

**10-Split-CIFAR100**

**For CIL (Class Incremental Learning) Settings:**

Change `"mode": "CIL"` in ```configs/cifar100_split.json``` file.
```
python main.py --config configs/cifar100_split.json
```

**For TIL (Task Incremental Learning) settings:**

Change `"mode": "TIL"` in ```configs/cifar100_split.json``` file.
```
python main.py --config configs/cifar100_split.json
```

**10-Split-ImageNet100**

Please change the code before training as:

1. PEGP/clip_pegp_prompt/datasets/data_manager.py

```
# class_data, class_targets = self._select(dataset, low_range=idx, high_range=idx + 1)
class_data, class_targets = self._select_img(dataset, low_range=idx, high_range=idx + 1)
```

2. PEGP/clip_pegp_prompt/models/tclip.py

```
# temp_names = list(cifar100_classnames.values())
temp_names = list(imagenet100_classnames.values())
```

**For CIL (Class Incremental Learning) Settings:**

Change `"mode": "CIL"` in ```configs/imagenet100_split.json``` file.
```
python main.py --config configs/imagenet100_split.json
```

**For TIL (Task Incremental Learning) settings:**

Change `"mode": "TIL"` in ```configs/imagenet100_split.json``` file.
```
python main.py --config configs/imagenet100_split.json
```

## Results
Training weights of 10-Split-CIFAR100 can be download in:

Link:https://pan.baidu.com/s/1tIy_k9gn1_rP7aXo2JpBXQ?pwd=ab3p Code:ab3p

