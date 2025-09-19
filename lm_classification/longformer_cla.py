import os
from tqdm import tqdm
from datetime import datetime

log_dir = "/media/luzhenyang/project/agent_graph_diag/lm_classification/training_logs"
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_txt_path = os.path.join(log_dir, f"log_{timestamp}.txt")
log_csv_path = os.path.join(log_dir, f"log_{timestamp}.csv")

csv_headers = ["epoch", "train_loss", "micro_f1", "macro_f1", "accuracy"]
with open(log_csv_path, "w") as f:
    f.write(",".join(csv_headers) + "\n")


from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

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

def log_metrics(epoch, train_loss, y_true, y_pred, log_txt_path, log_csv_path):
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

#------------------------------------------------------- 加载模型和Tokenizer
# Using tiantanvte env
from transformers import BigBirdForSequenceClassification, BigBirdTokenizer, BigBirdConfig
# LongformerTokenizer, LongformerForSequenceClassification, LongformerConfig

model_name = 'l-yohai/bigbird-roberta-base-mnli'
# "allenai/longformer-base-4096"

# 加载 tokenizer 和模型配置
tokenizer = BigBirdTokenizer.from_pretrained(model_name)
config = BigBirdConfig.from_pretrained(model_name)
config.model_type = "bigbird"
config.num_labels = 4

# 加载模型
model = BigBirdForSequenceClassification.from_pretrained(
    model_name, 
    config=config,
    trust_remote_code=True,
    use_safetensors=True
)



#------------------------------------------------------- 构建数据集和数据加载器
import pandas as pd
from utils import AcuteAbdominalDiagnosisDataset, get_dataloader
from sklearn.model_selection import train_test_split

df = pd.read_csv("/media/luzhenyang/project/agent_graph_diag/lm_classification/ab_cls_dataset.csv")
label_list = df['diagnosis'].unique().tolist()
print("label_list: ", label_list)

# Tokenizer + Padding + Truncation 处理
encoded = [
    tokenizer(
        text, 
        padding='max_length',
        truncation=True,
        max_length=4096,
        # return_tensors='pt' 会导致 input_ids shape: torch.Size([4, 1, 4096])
    ) for text in tqdm(df['context'].tolist())
]

df['input_ids'] = [ e['input_ids'] for e in encoded ]
df['attention_mask'] = [ e['attention_mask'] for e in encoded ]
df['input_length'] = df['context'].apply(lambda x: len(tokenizer.tokenize(x)))

print(df.shape)

# 划分训练集，测试集
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=7)

print(train_df['diagnosis'].value_counts())
print(val_df['diagnosis'].value_counts())

# 列：['context', 'diagnosis']
train_dataset = AcuteAbdominalDiagnosisDataset(train_df)
val_dataset = AcuteAbdominalDiagnosisDataset(val_df)
train_loader = get_dataloader(train_dataset, use_weighted=True)
val_loader = get_dataloader(val_dataset, use_weighted=False)



#------------------------------------------------------- 模型训练逻辑（基本版）
from torch.optim import AdamW
from tqdm import tqdm
import torch
from utils import seed_torch
from sklearn.metrics import classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_torch(device=device)
model.to(device)

# ---------- 配置 ----------
MAX_LENGTH = 4096
BATCH_SIZE = 4
NUM_EPOCHS = 3
LR = 2e-5

optimizer = AdamW(model.parameters(), lr=LR)

# ---------- 训练函数 ----------
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1

        # print("input_ids shape:", input_ids.shape)
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            global_attention_mask=global_attention_mask, 
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)

# ---------- 验证函数 ----------
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    report = classification_report(all_labels, all_preds, target_names=label_list, digits=4)
    return report

# ---------- 主训练循环 ----------
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    train_loss = train_epoch(model, train_loader, optimizer)
    print(f"Train loss: {train_loss:.4f}")
    
    # 验证并获取预测与标签
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # 打印+写入日志
    log_metrics(epoch, train_loss, all_labels, all_preds, log_txt_path, log_csv_path)

    # 保存模型
    save_path = f"checkpoint_epoch{epoch+1}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


