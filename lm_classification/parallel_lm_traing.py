# train_one_seed.py
import os
import torch
import pandas as pd
import argparse
from datetime import datetime
from tqdm import tqdm
from transformers import LongformerTokenizer
from utils import AcuteAbdominalDiagnosisDataset, get_dataloader, setup_model_and_optimizer, train_epoch, log_metrics
from sklearn.metrics import accuracy_score

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

def main(seed, device_id):
    # ---------- 设置设备 ----------
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # ---------- 加载数据 ----------
    model_name = "allenai/longformer-base-4096" # "yikuan8/Clinical-Longformer"
    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    df = pd.read_csv("/media/luzhenyang/project/agent_graph_diag/lm_classification/ab_cls_dataset_v2_complete_info.csv")
    label_list = df['diagnosis'].unique().tolist()

    encoded = [
        tokenizer(text, padding='max_length', truncation=True, max_length=4096)
        for text in tqdm(df['context'].tolist())
    ]
    df['input_ids'] = [e['input_ids'] for e in encoded]
    df['attention_mask'] = [e['attention_mask'] for e in encoded]

    val_ids = pd.read_csv(f"/media/luzhenyang/project/agent_graph_diag/subset_ids_{seed}.csv")
    val_df = df[df['hadm_id'].isin(val_ids['hadm_id'].values)].reset_index(drop=True)
    train_df = df[~df['hadm_id'].isin(val_ids['hadm_id'].values)].reset_index(drop=True)

    train_dataset = AcuteAbdominalDiagnosisDataset(train_df)
    val_dataset = AcuteAbdominalDiagnosisDataset(val_df)
    train_loader = get_dataloader(train_dataset, use_weighted=True)
    val_loader = get_dataloader(val_dataset, use_weighted=False)

    model, tokenizer, optimizer, scheduler = setup_model_and_optimizer(train_loader, device=device)
    model.to(device)

    log_dir = f"/media/luzhenyang/project/agent_graph_diag/lm_classification/training_logs_normal_longformer_rs"
    os.makedirs(log_dir, exist_ok=True)
    log_txt_path = os.path.join(log_dir, f"log_seed{seed}.txt")
    log_csv_path = os.path.join(log_dir, f"log_seed{seed}.csv")

    sample_level_records = []

    for epoch in range(3):
        print(f"\n[Seed {seed}] Epoch {epoch+1}/3")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}")

        # 评估
        model.eval()
        all_preds, all_labels, all_hadm_ids = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
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
                all_hadm_ids.extend(batch['hadm_id'])

        log_metrics(epoch, train_loss, all_labels, all_preds, log_txt_path, log_csv_path, label_list)

        for hid, label, pred in zip(all_hadm_ids, all_labels, all_preds):
            sample_level_records.append({"hadm_id": hid, "label": label, "pred": pred, "seed": seed})

        # 保存模型
        model_path = os.path.join(log_dir, f"checkpoint_epoch{epoch+1}_seed{seed}.pt")
        torch.save(model.state_dict(), model_path)

    # 保存样本级别预测结果
    df_result = pd.DataFrame(sample_level_records)
    df_result.to_csv(os.path.join(log_dir, f"predictions_seed_{seed}.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    main(args.seed, args.gpu)
