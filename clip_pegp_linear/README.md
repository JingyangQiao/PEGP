## Usage

First, clone our repository:

```
git clone https://github.com/JingyangQiao/PEGP
cd PEGP/clip_pegp_linear
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
Experimental results of each task are provided in ```test_stats.txt``` file.

Training weights can be obtained by contacting via email.

