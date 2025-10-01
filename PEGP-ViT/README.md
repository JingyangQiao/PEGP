## Usage

First, clone our repository:

```
git clone https://github.com/JingyangQiao/PEGP
cd PEGP/PEGP-ViT
```

Second, install the conda environment by running:

```
conda env create -f environment.yaml -n pegp
```

Third, activate the conda environment by running:

```
conda activate pegp
```

## Data preparation

If you already have CIFAR-100 or ImageNet-R or Domainnet, pass your dataset path to  `.data.root` in each 
```config/base/*.yaml``` file, namely ```config/base/cifar100_order1.yaml```, ```config/base/imagenet-r_order1.yaml``` and ```config/base/domainnet_order1.yaml```.

## Training

To train a model via command line:

**10-Split-CIFAR100 (Adapter)**

Change extends in ```config/vit_adapter.yaml``` to ```./base/cifar100_order1.yaml```

```
python main.py --config vit_adapter.json
```

**10-Split-CIFAR100 (LoRA)**

Change extends in ```config/vit_lora.yaml``` to ```./base/cifar100_order1.yaml```

```
python main.py --config vit_lora.json
```

**10-Split-ImageNet-R (Adapter)**

Change extends in ```config/vit_adapter.yaml``` to ```./base/imagenet-r_order1.yaml```

```
python main.py --config vit_adapter.json
```

**10-Split-ImageNet-R (LoRA)**

Change extends in ```config/vit_lora.yaml``` to ```./base/imagenet-r_order1.yaml```

```
python main.py --config vit_lora.json
```

**5-Split-DomainNet (Adapter)**

Change extends in ```config/vit_adapter.yaml``` to ```./base/domainnet_order1.yaml```

```
python main.py --config vit_adapter.json
```

**5-Split-DomainNet (LoRA)**

Change extends in ```config/vit_lora.yaml``` to ```./base/domainnet_order1.yaml```

```
python main.py --config vit_lora.json
```

## Thanks

The baseline code of LAE from "A unified continual learning framework with general parameter efficient tuning".


## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
