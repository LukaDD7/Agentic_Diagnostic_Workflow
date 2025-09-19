import os
import math
import torch
import pickle
import random
import numpy as np
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from transformers import (
    LongformerForSequenceClassification,
    LongformerConfig,
    LongformerTokenizer,
    get_linear_schedule_with_warmup
)


# label_list = ['appendicitis', 'cholecystitis', 'pancreatitis', 'diverticulitis']

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)
        

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def seed_torch(device, seed=7):
    # ------------------------------------------------------------------------------------------
    # References:
    # HIPT: https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/main.py
    # ------------------------------------------------------------------------------------------
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AcuteAbdominalDiagnosisDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ):
        """
        Args:
            df: The DataFrame of the csv file dataset
            column "context" represent the X
            column "diagnosis" represent the y but still a str label
        """
        self.input_ids = df['input_ids'].tolist()
        self.attention_mask = df['attention_mask'].tolist()
        self.labels = df['diagnosis'].tolist()
        self.hadm_ids = df['hadm_id'].tolist()
        label2id = {
            'appendicitis': 0,
            'cholecystitis': 1,
            'pancreatitis': 2,
            'diverticulitis': 3
        }
        self.labels = [label2id[label] for label in self.labels]


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx]).clone().detach(),
            'attention_mask': torch.tensor(self.attention_mask[idx]).clone().detach(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'hadm_id': self.hadm_ids[idx],
        }

def get_dataloader(dataset: AcuteAbdominalDiagnosisDataset, batch_size=6, use_weighted=False):
    """根据 use_weighted 决定是否使用过采样"""
    if use_weighted:
        label_counts = Counter(dataset.labels)
        num_samples = len(dataset.labels)

        # 为每个类别计算权重: 样本越少，权重越大
        class_weights = {
            cls: num_samples / count for cls, count in label_counts.items()
        }

        # 为每个样本分配权重
        sample_weights = [class_weights[label] for label in dataset.labels]

        # 构造 WeightedRandomSampler
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)

        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        # 验证集或测试集：不使用采样，顺序加载或打乱都可以
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', device=None):
        super(FocalLoss, self).__init__()
        if alpha is not None and device is not None:
            self.alpha = alpha.to(device)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # pt 是正确类别的预测概率
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1

        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            global_attention_mask=global_attention_mask, 
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()  # 更新学习率
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)

def per_class_accuracy(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    total = cm.sum()
    per_class_acc = {}
    for i, cls in enumerate(class_names):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = total - TP - FP - FN
        accuracy_i = (TP + TN) / total
        per_class_acc[cls] = accuracy_i
    return per_class_acc

def log_metrics(epoch, train_loss, y_true, y_pred, log_txt_path, log_csv_path, label_list):
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    # 计算每个类别准确率
    per_class_acc = per_class_accuracy(y_true, y_pred, label_list)

    # 文本日志
    with open(log_txt_path, "a") as f:
        f.write(f"\nEpoch {epoch+1}\n")
        f.write(f"Train Loss: {train_loss:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Micro F1: {micro_f1:.4f} | Macro F1: {macro_f1:.4f}\n")

        # 每个类别准确率写入日志
        for cls, acc_cls in per_class_acc.items():
            f.write(f"Accuracy for {cls}: {acc_cls:.4f}\n")

        f.write(f"{classification_report(y_true, y_pred, target_names=label_list, digits=4)}\n")

    # CSV日志
    with open(log_csv_path, "a") as f:
        f.write(f"{epoch+1},{train_loss:.4f},{micro_f1:.4f},{macro_f1:.4f},{acc:.4f}\n")

def setup_model_and_optimizer(train_loader,
                              model_name="yikuan8/Clinical-Longformer",
                              num_labels=4,
                              lr=2e-5,
                              weight_decay=0.01,
                              num_epochs=3,
                              freeze_except_last=True,
                              device=None):
    """
    初始化 Longformer 模型、优化器与学习率调度器。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载 tokenizer 和 config
    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    config = LongformerConfig.from_pretrained(model_name)
    config.num_labels = num_labels

    # 2. 加载模型
    model = LongformerForSequenceClassification.from_pretrained(model_name, config=config)

    # 3. 冻结部分层（可选）
    if freeze_except_last:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any([
                name.startswith("longformer.encoder.layer.10"),
                name.startswith("longformer.encoder.layer.11"),
                name.startswith("classifier"),
                "LayerNorm" in name,
            ]):
                param.requires_grad = True

    model.to(device)

    # 4. 设置优化器
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # 5. 设置调度器
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 打印信息
    print(f"[✓] Loaded model: {model_name}")
    print(f"[✓] Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"[✓] Total steps: {num_training_steps}, Warmup: {num_warmup_steps}")

    return model, tokenizer, optimizer, scheduler