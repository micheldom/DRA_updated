import os
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score

from dataloaders.dataloader import initDataloader
from modeling.net import DRA
from modeling.layers import build_criterion

WEIGHT_DIR = Path('./weights')

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        
        kwargs = {'num_workers': args.workers}
        self.train_loader, self.test_loader = initDataloader.build(args, **kwargs)
        
        if self.args.total_heads == 4:
            temp_args = args
            temp_args.batch_size = self.args.nRef
            temp_args.nAnomaly = 0
            self.ref_loader, _ = initDataloader.build(temp_args, **kwargs)
            self.ref = iter(self.ref_loader)

        self.model = DRA(args, backbone=self.args.backbone).to(self.device)

        if self.args.pretrain_dir:
            self.model.load_state_dict(torch.load(self.args.pretrain_dir))
            print(f'Load pretrain weight from: {self.args.pretrain_dir}')

        self.criterion = build_criterion(args.criterion).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def generate_target(self, target, eval=False):
        targets = []
        if eval:
            targets.extend([target == 0, target, target, target])
        else:
            temp_t = target != 0
            targets.extend([
                target == 0,
                temp_t[target != 2],
                temp_t[target != 1],
                target != 0
            ])
        return targets

    def training(self, epoch):
        self.model.train()
        train_loss = 0.0
        class_loss = [0.0] * self.args.total_heads
        tbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for sample in tbar:
            image, target = sample['image'].to(self.device), sample['label'].to(self.device)

            if self.args.total_heads == 4:
                try:
                    ref_image = next(self.ref)['image'].to(self.device)
                except StopIteration:
                    self.ref = iter(self.ref_loader)
                    ref_image = next(self.ref)['image'].to(self.device)
                image = torch.cat([ref_image, image], dim=0)

            outputs = self.model(image, target)
            targets = self.generate_target(target)

            losses = [
                self.criterion(F.softmax(outputs[i], dim=1), targets[i].long()) if self.args.criterion == 'CE'
                else self.criterion(outputs[i], targets[i].float())
                for i in range(self.args.total_heads)
            ]

            loss = torch.sum(torch.cat([l.view(-1, 1) for l in losses]))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            train_loss += loss.item()
            for i in range(self.args.total_heads):
                class_loss[i] += losses[i].item()

            tbar.set_postfix(train_loss=train_loss / (len(tbar) + 1))

    def normalization(self, data):
        return data

    def eval(self):
        self.model.eval()
        test_loss = 0.0
        class_pred = [np.array([]) for _ in range(self.args.total_heads)]
        total_target = np.array([])

        tbar = tqdm(self.test_loader, desc='Evaluating')

        with torch.no_grad():
            for sample in tbar:
                image, target = sample['image'].to(self.device), sample['label'].to(self.device)

                if self.args.total_heads == 4:
                    try:
                        ref_image = next(self.ref)['image'].to(self.device)
                    except StopIteration:
                        self.ref = iter(self.ref_loader)
                        ref_image = next(self.ref)['image'].to(self.device)
                    image = torch.cat([ref_image, image], dim=0)

                outputs = self.model(image, target)
                targets = self.generate_target(target, eval=True)

                losses = [
                    self.criterion(F.softmax(outputs[i], dim=1), targets[i].long()) if self.args.criterion == 'CE'
                    else self.criterion(outputs[i], targets[i].float())
                    for i in range(self.args.total_heads)
                ]

                loss = sum(losses)
                test_loss += loss.item()
                tbar.set_postfix(test_loss=test_loss / (len(tbar) + 1))

                for i in range(self.args.total_heads):
                    data = -1 * outputs[i].data.cpu().numpy() if i == 0 else outputs[i].data.cpu().numpy()
                    class_pred[i] = np.append(class_pred[i], data)
                total_target = np.append(total_target, target.cpu().numpy())

        total_pred = self.normalization(class_pred[0])
        for i in range(1, self.args.total_heads):
            total_pred += self.normalization(class_pred[i])

        results_path = Path(self.args.experiment_dir) / 'result.txt'
        with results_path.open('a+', encoding="utf-8") as w:
            for label, score in zip(total_target, total_pred):
                w.write(f'{label}   {score}\n')

        total_roc, total_pr = aucPerformance(total_pred, total_target)

        normal_mask = total_target == 0
        outlier_mask = total_target == 1
        plt.clf()
        plt.bar(np.arange(total_pred.size)[normal_mask], total_pred[normal_mask], color='green')
        plt.bar(np.arange(total_pred.size)[outlier_mask], total_pred[outlier_mask], color='red')
        plt.ylabel("Anomaly score")
        plt.savefig(Path(self.args.experiment_dir) / "vis.png")
        return total_roc, total_pr

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), Path(self.args.experiment_dir) / filename)

    def load_weights(self, filename):
        self.model.load_state_dict(torch.load(Path(WEIGHT_DIR) / filename))

    def init_network_weights_from_pretraining(self):
        net_dict = self.model.state_dict()
        ae_net_dict = self.ae_model.state_dict()
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        net_dict.update(ae_net_dict)
        self.model.load_state_dict(net_dict)

def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print(f"AUC-ROC: {roc_auc:.4f}, AUC-PR: {ap:.4f}")
    return roc_auc, ap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=48, help="batch size used in SGD")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--cont_rate", type=float, default=0.0, help="outlier contamination rate in training data")
    parser.add_argument("--test_threshold", type=int, default=0, help="outlier contamination rate in training data")
    parser.add_argument("--test_rate", type=float, default=0.0, help="outlier contamination rate in training data")
    parser.add_argument("--dataset", type=str, default='mvtecad', help="dataset name")
    parser.add_argument("--ramdn_seed", type=int, default=42, help="random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--savename', type=str, default='model.pkl', help="save model name")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiment/experiment_14', help="experiment directory")
    parser.add_argument('--classname', type=str, default='capsule', help="dataset class")
    parser.add_argument('--img_size', type=int, default=448, help="image size")
    parser.add_argument("--nAnomaly", type=int, default=10, help="number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='resnet18', help="backbone model")
    parser.add_argument('--criterion', type=str, default='deviation', help="loss criterion")
    parser.add_argument("--topk", type=float, default=0.1, help="topk in MIL")
    parser.add_argument('--know_class', type=str, default=None, help="set the known class for hard setting")
    parser.add_argument('--pretrain_dir', type=str, default=None, help="pretrain weight directory")
    parser.add_argument("--total_heads", type=int, default=4, help="number of heads in training")
    parser.add_argument("--nRef", type=int, default=5, help="number of reference set")
    parser.add_argument('--outlier_root', type=str, default=None, help="OOD dataset root")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    trainer = Trainer(args)

    argsDict = args.__dict__
    experiment_dir = Path(args.experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    with (experiment_dir / 'setting.txt').open('w') as f:
        f.write('------------------ start ------------------\n')
        for eachArg, value in argsDict.items():
            f.write(f'{eachArg} : {value}\n')
        f.write('------------------- end -------------------')

    print(f'Total Epochs: {trainer.args.epochs}')
    
    for epoch in range(trainer.args.epochs):
        trainer.training(epoch)
    
    trainer.eval()
    trainer.save_weights(args.savename)