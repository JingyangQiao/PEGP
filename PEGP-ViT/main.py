# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from absl import logging, flags

from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from sklearn.decomposition import PCA

from libml.core import App, Conf, DataModule, Module, ModuleConf
from libml.data import build_imagenet_transform
from libml.model.pet import Adapter, Conv2dAdapter, LinearLoRA, Prefix, Prompt
from libml.model.utils import freeze, unfreeze
from libml.model.backbone.swin import SwinTransformer
from libml.model.backbone.convnext import ConvNeXt


@dataclass
class MyModuleConf(ModuleConf):
    adapt_blocks: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    pet_cls: str = "Adapter"  # enum: Adapter, LoRA, Prefix
    pet_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
    num_freeze_epochs: int = 3



@dataclass
class MyConf(Conf):
    module: MyModuleConf = MyModuleConf()


class MyDataModule(DataModule):
    def train_transform(self):
        return build_imagenet_transform(train=True, norm=False)

    def val_transform(self):
        return build_imagenet_transform(train=False, norm=False)


class MyModule(Module):
    pets: nn.Module  # online pet modules
    pets_: nn.ModuleList  # offline pet modules
    original_vit: nn.Module
    train_acc: Accuracy
    loss_fn: nn.Module

    def setup_model(self):
        super().setup_model()

        if getattr(self, "pets_", None) is None:
            freeze(self.model.backbone)
            self.pets = self.create_pets()
            self.pets_ = self.create_pets()
            # logging.info(f"==> pets:\n{self.pets}")

        self.train_acc = Accuracy()
        self.loss_fn = nn.CrossEntropyLoss()

        self.attach_pets(self.pets)

        if getattr(self, "feature_adapter", None) is None:
            self.feature_adapter = {}
            print("Reset Feature Matrix!!")
        # for n, p in self.named_parameters():
        #     print(n, p.size())
        # if getattr(self, "feature_prompt", None) is None:
        #     self.feature_prompt = {}
        #     print("Reset Feature Matrix!!")

    def create_pets_swin(self):
        assert self.cfg.pet_cls == "Adapter", "Not implemented PET"

        blocks = []
        for layer in self.model.backbone.layers:
            blocks += list(layer.blocks.children())

        kwargs = self.cfg.pet_kwargs
        adapters = [
            Adapter(blocks[idx].dim, **kwargs) for idx in self.cfg.adapt_blocks
        ]
        return nn.ModuleList(adapters)

    def create_pets_convnext(self):
        assert self.cfg.pet_cls == "Adapter", "Not implemented PET"

        n = len(self.cfg.adapt_blocks)
        stages = self.model.backbone.stages
        blocks = [list(stage.blocks.children()) for stage in stages.children()]
        blocks = sum(blocks, [])
        adapters = []
        for idx in self.cfg.adapt_blocks:
            dim = blocks[idx].conv_dw.in_channels
            adapter = Conv2dAdapter(dim, dim, **self.cfg.pet_kwargs)
            adapters.append(adapter)
        return nn.ModuleList(adapters)

    def create_pets_vit(self):
        assert self.cfg.pet_cls in ["Adapter", "LoRA", "Prefix", "Prompt"]

        n = len(self.cfg.adapt_blocks)
        embed_dim = self.model.backbone.embed_dim

        kwargs = dict(**self.cfg.pet_kwargs)
        if self.cfg.pet_cls == "Adapter":
            kwargs["embed_dim"] = embed_dim
            return nn.ModuleList([Adapter(**kwargs) for _ in range(n)])

        if self.cfg.pet_cls == "LoRA":
            kwargs["in_features"] = embed_dim
            kwargs["out_features"] = embed_dim * 3
            return nn.ModuleList([LinearLoRA(**kwargs) for _ in range(n)])

        if self.cfg.pet_cls == "Prompt":
            kwargs["dim"] = embed_dim
            return nn.ModuleList([Prompt(**kwargs) for i in range(n)])

        kwargs["dim"] = embed_dim
        return nn.ModuleList([Prefix(**kwargs) for i in range(n)])

    def create_pets(self):
        if isinstance(self.model.backbone, SwinTransformer):
            return self.create_pets_swin()

        if isinstance(self.model.backbone, ConvNeXt):
            return self.create_pets_convnext()

        return self.create_pets_vit()

    def attach_pets_swin(self, pets: nn.ModuleList):
        assert self.cfg.pet_cls == "Adapter", "Not implemented PET"

        blocks = []
        for layer in self.model.backbone.layers:
            blocks += list(layer.blocks.children())

        for i, b in enumerate(self.cfg.adapt_blocks):
            blocks[b].attach_adapter(attn=pets[i])

    def attach_pets_convnext(self, pets: nn.ModuleList):
        assert self.cfg.pet_cls == "Adapter", "Not implemented PET"

        n = len(self.cfg.adapt_blocks)
        stages = self.model.backbone.stages
        blocks = [
            [(idx, stage) for idx, _ in enumerate(stage.blocks.children())]
            for stage in stages.children()
        ]
        blocks = sum(blocks, [])
        for i, b in enumerate(self.cfg.adapt_blocks):
            idx, stage = blocks[b]
            stage.attach_adapter(**{f"blocks.{idx}": pets[i]})

    def attach_pets_vit(self, pets: nn.ModuleList):
        assert self.cfg.pet_cls in ["Adapter", "LoRA", "Prefix", "Prompt"]

        if self.cfg.pet_cls == "Adapter":
            for i, b in enumerate(self.cfg.adapt_blocks):
                self.model.backbone.blocks[b].attach_adapter(attn=pets[i])
                self.model.backbone.use_adapter = True
            return

        if self.cfg.pet_cls == "LoRA":
            for i, b in enumerate(self.cfg.adapt_blocks):
                self.model.backbone.blocks[b].attn.attach_adapter(qkv=pets[i])
                self.model.backbone.use_lora = True
            return

        if self.cfg.pet_cls == "Prefix":
            for i, b in enumerate(self.cfg.adapt_blocks):
                self.model.backbone.blocks[b].attn.attach_prefix(pets[i])
                self.model.backbone.use_prefix = True
            return

        if self.cfg.pet_cls == "Prompt":
            for i, b in enumerate(self.cfg.adapt_blocks):
                self.model.backbone.blocks[b].attn.attach_prompt(pets[i])
                self.model.backbone.use_prompt = True
            return

    def attach_pets(self, pets: nn.ModuleList):
        if isinstance(self.model.backbone, SwinTransformer):
            return self.attach_pets_swin(pets)

        if isinstance(self.model.backbone, ConvNeXt):
            return self.attach_pets_convnext(pets)

        return self.attach_pets_vit(pets)

    def filter_state_dict(self, state_dict):
        ps = ("model_wrap.", "model.backbone.")
        return {k: v for k, v in state_dict.items() if not k.startswith(ps)}

    def forward(self, input):
        return super().forward(input)

    def pre_train_epoch(self):
        self.train_acc.reset()

        if self.current_task == 0 or self.cfg.num_freeze_epochs < 1:
            return

        if self.current_epoch == 0:
            freeze(self.pets)
            logging.info("===> Freeze pets")

        if self.current_epoch == self.cfg.num_freeze_epochs:
            unfreeze(self.pets)
            logging.info("===> Unfreeze pets")

    def train_step(
        self, batch, batch_idx
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        num_old_classes = self.datamodule.num_old_classes
        output = self(batch[0])
        self.train_acc.update(output, batch[1])
        output = output[:, num_old_classes:]
        target = batch[1] - num_old_classes
        loss = self.loss_fn(output, target)
        loss_dict = {"loss": loss, "acc": self.train_acc.compute() * 100}
        return loss, loss_dict

    def train_step_(
        self, batch, batch_idx
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        num_old_classes = self.datamodule.num_old_classes

        self.attach_pets(self.pets)
        output = self(batch[0])
        self.train_acc.update(output, batch[1])
        output = output[:, num_old_classes:]
        target = batch[1] - num_old_classes
        loss = self.loss_fn(output, target)

        self.attach_pets(self.pets_)
        fa = self.model.backbone(batch[0])
        self.attach_pets(self.pets)
        fb = self.model.backbone(batch[0])
        loss += (fa - fb).pow(2).mean()

        loss_dict = {"loss": loss, "acc": self.train_acc.compute() * 100}
        return loss, loss_dict

    def post_train_step(self):
        pass

    def eval_epoch(self, loader, train_loader):
        task_ranges = []
        n_tasks = self.current_task + 1
        for t in range(n_tasks):
            s = task_ranges[-1][-1] + 1 if task_ranges else 0
            e = s + self.datamodule.num_classes_of(t)
            task_ranges.append(list(range(s, e)))

        pred_on, pred_off, pred_ens, gts = [], [], [], []

        self.attach_pets(self.pets_)
        for _, batch in enumerate(loader):
            input, target = batch[:2]
            output, bs = self(input), input.shape[0]
            pred_on.append(output.argmax(dim=1))
            output_ = [output.softmax(dim=1)]

            for oe in output_[1:]:
                pred_off.append(oe.argmax(dim=1))
                break

            output = torch.stack(output_, dim=-1).max(dim=-1)[0]
            self.val_acc.update(output, target)
            for t in batch[2].long().unique().tolist():
                sel = batch[2] == t
                self.val_task_accs[t].update(output[sel], target[sel])
                t_range = task_ranges[t]
                output_local = output[sel][:, t_range]
                target_local = target[sel] - t_range[0]
                self.val_task_local_accs[t].update(output_local, target_local)

            pred_ens.append(output.argmax(dim=1))
            gts.append(target)

        adapter_rep = self.get_adapter_matrix(train_loader)
        if self.cfg.pet_cls == "Adapter" or self.cfg.pet_cls == "LoRA":
            if self.cfg.pet_cls == "Adapter":
                self.threshold = 0.999
            else:
                self.threshold = 0.99
            self.update_memory_adapter(adapter_rep)

            self.feature_t = {0: {"down": [], "up": []}, 1: {"down": [], "up": []}, 2: {"down": [], "up": []},
                              3: {"down": [], "up": []}, 4: {"down": [], "up": []}}

            for layer in self.feature_adapter:
                for item in self.feature_adapter[layer]:
                    temp_feature = self.feature_adapter[layer][item].reshape(self.feature_adapter[layer][item].shape[0], -1)
                    Uf = torch.Tensor(np.dot(temp_feature, temp_feature.transpose())).cuda()
                    print(layer, item, Uf.size())
                    self.feature_t[layer][item] = Uf
        else:
            raise NotImplementedError

        return pred_on, pred_off, pred_ens, gts

    def post_eval_epoch(self, result):
        super().post_eval_epoch()
        if self.current_task == 0 or not flags.FLAGS.debug:
            return

        pred_on, pred_off, pred_ens, gts = result
        pred_on = torch.cat(pred_on)
        pred_off = torch.cat(pred_off) if len(pred_off) else pred_on[:0]
        pred_ens = torch.cat(pred_ens)
        gts = torch.cat(gts)
        torch.save(pred_on, f=f"{self.trainer.ckpt_dir}/pred_on.pt")
        torch.save(pred_off, f=f"{self.trainer.ckpt_dir}/pred_off.pt")
        torch.save(pred_ens, f=f"{self.trainer.ckpt_dir}/pred_ens.pt")
        torch.save(gts, f=f"{self.trainer.ckpt_dir}/gts.pt")

    def configure_optimizer(self, *_):
        return super().configure_optimizer(self.model.head, self.pets)

    def get_adapter_matrix(self, loader):
        count = 1
        representation = {}

        with torch.no_grad():
            for _, batch in enumerate(loader):
                input, target = batch[:2]
                input = input.cuda()
                output = self(input)
                for layer in range(5):
                    if layer not in representation:
                        representation[layer] = {"down": [], "up": []}
                    representation[layer]["down"].append(self.model.backbone.collect[layer]["down"])
                    representation[layer]["up"].append(self.model.backbone.collect[layer]["up"])
                count += 1
                if count > 32:  # 选取sample的数量
                    for layer in representation:
                        for item in representation[layer]:
                            representation[layer][item] = torch.cat(representation[layer][item])
                            representation[layer][item] = representation[layer][item].detach().cpu().numpy()
                            representation[layer][item] = representation[layer][item].reshape(representation[layer][item].shape[0], -1)
                            rep = representation[layer][item]

                            if item == "down":
                                pca = PCA(n_components=768)
                            else:
                                try:
                                    pca = PCA(n_components=self.cfg.pet_kwargs.down_sample)
                                except:
                                    pca = PCA(n_components=self.cfg.pet_kwargs.rank)
                            pca = pca.fit(rep)
                            rep = pca.transform(rep)
                            representation[layer][item] = rep
                            print(layer, item, representation[layer][item].shape)

                    break

        return representation

    def update_memory_adapter(self, represent):
        for layer in represent:
            for item in represent[layer]:
                representation = represent[layer][item]
                representation = np.matmul(representation.T, representation)
                try:
                    feature = self.feature_adapter[layer][item]
                except:
                    feature = None
                if feature is None:
                    if layer not in self.feature_adapter:
                        self.feature_adapter[layer] = {}
                    U, S, Vh = np.linalg.svd(representation, full_matrices=False)
                    sval_total = (S ** 2).sum()
                    sval_ratio = (S ** 2) / sval_total
                    r = np.sum(np.cumsum(sval_ratio) < self.threshold)
                    feature = U[:, 0:r]
                else:
                    U1, S1, Vh1 = np.linalg.svd(representation, full_matrices=False)
                    sval_total = (S1 ** 2).sum()
                    # Projected Representation
                    act_hat = representation - np.dot(np.dot(feature, feature.transpose()), representation)
                    U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria
                    sval_hat = (S ** 2).sum()
                    sval_ratio = (S ** 2) / sval_total
                    accumulated_sval = (sval_total - sval_hat) / sval_total
                    r = 0
                    for ii in range(sval_ratio.shape[0]):
                        if accumulated_sval < self.threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        feature = feature
                    # update GPM
                    U = np.hstack((feature, U[:, 0:r]))
                    feature = U
                print('-' * 40)
                print('Gradient Constraints Summary', feature.shape)
                print('-' * 40)
                self.feature_adapter[layer][item] = feature


if __name__ == "__main__":
    kwargs = dict(
        datamodule_cls=MyDataModule, module_cls=MyModule, config_cls=MyConf
    )
    App(**kwargs).run()
