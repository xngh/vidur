import argparse
import json
import logging
import random
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from vidur.predictor.workload_profiler import WorkloadProfiler, WorkloadProfilerConfig

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkloadDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        task_type = item['task_type']
        label = item['label']
        prompt_len = item.get('prompt_len')
        if prompt_len is None:
            prompt_len = len(str(text).split())

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'task_type_ids': torch.tensor(task_type, dtype=torch.long),
            'prompt_len': torch.tensor(prompt_len, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def train_val_split(
    data: List[Dict], val_ratio: float, seed: int
) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    split = int(len(indices) * (1 - val_ratio))
    train_idx = indices[:split]
    val_idx = indices[split:]
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    return train_data, val_data


def compute_metrics(preds: List[int], labels: List[int], num_classes: int) -> Dict[str, float]:
    if len(preds) != len(labels):
        raise ValueError("preds and labels must have the same length")

    total = len(labels)
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    accuracy = correct / total if total > 0 else 0.0

    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for p, y in zip(preds, labels):
        if p == y:
            tp[y] += 1
        else:
            fp[p] += 1
            fn[y] += 1

    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    for c in range(num_classes):
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)

    macro_precision = sum(precision_per_class) / num_classes if num_classes > 0 else 0.0
    macro_recall = sum(recall_per_class) / num_classes if num_classes > 0 else 0.0
    macro_f1 = sum(f1_per_class) / num_classes if num_classes > 0 else 0.0

    # Balanced accuracy is macro recall
    balanced_accuracy = macro_recall

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
    }


def evaluate_model(model, dataloader, device, num_classes: int) -> Dict[str, float]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            task_type_ids = batch["task_type_ids"].to(device)
            labels = batch["labels"].to(device)

            prompt_len = batch["prompt_len"].to(device)
            logits = model(input_ids, attention_mask, task_type_ids, prompt_len=prompt_len)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return compute_metrics(all_preds, all_labels, num_classes)


def train_model(args):
    # 1. 加载已生成的数据集 (JSONL)
    logger.info("Loading dataset...")
    processed_data = load_jsonl(args.dataset_path)
    if not processed_data:
        raise ValueError(f"Dataset is empty: {args.dataset_path}")

    # 1.5 校验并对齐数据分布与配置
    labels = [item.get("label") for item in processed_data]
    task_types = [item.get("task_type") for item in processed_data]
    if any(l is None for l in labels):
        raise ValueError("Dataset has missing 'label' fields.")
    if any(t is None for t in task_types):
        raise ValueError("Dataset has missing 'task_type' fields.")

    min_label = min(labels)
    max_label = max(labels)
    if min_label < 0:
        raise ValueError(f"Found negative labels in dataset: min_label={min_label}")
    if args.num_buckets <= max_label:
        inferred_buckets = max_label + 1
        logger.warning(
            "num_buckets (%d) is too small for labels (max=%d). "
            "Auto-adjusting num_buckets to %d.",
            args.num_buckets,
            max_label,
            inferred_buckets,
        )
        args.num_buckets = inferred_buckets

    min_task_type = min(task_types)
    max_task_type = max(task_types)
    if min_task_type < 0:
        raise ValueError(f"Found negative task_type values: min_task_type={min_task_type}")
    if args.num_task_types <= max_task_type:
        inferred_task_types = max_task_type + 1
        logger.warning(
            "num_task_types (%d) is too small for task_type_ids (max=%d). "
            "Auto-adjusting num_task_types to %d.",
            args.num_task_types,
            max_task_type,
            inferred_task_types,
        )
        args.num_task_types = inferred_task_types

    # 2. 初始化配置和模型
    config = WorkloadProfilerConfig(
        model_name=args.model_name,
        num_buckets=args.num_buckets,
        num_task_types=args.num_task_types,
        freeze_bert=args.freeze_bert,
        unfreeze_bert_layers=args.unfreeze_bert_layers,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = WorkloadProfiler(config)
    
    # 3. 数据集准备 (train/val split)
    train_data, val_data = train_val_split(
        processed_data, val_ratio=args.val_ratio, seed=args.seed
    )
    train_dataset = WorkloadDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = WorkloadDataset(val_data, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 4. 优化器与损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 5. 训练循环
    logger.info("Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unknown"
        logger.info(
            "Using device: %s | num_gpus=%d | gpu0=%s | torch_cuda=%s",
            device,
            torch.cuda.device_count(),
            gpu_name,
            getattr(torch.version, "cuda", None),
        )
    else:
        logger.info("Using device: %s (CUDA not available)", device)
    model.to(device)
    
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            task_type_ids = batch['task_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            prompt_len = batch["prompt_len"].to(device)
            logits = model(input_ids, attention_mask, task_type_ids, prompt_len=prompt_len)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / max(1, len(train_loader))
        val_metrics = evaluate_model(model, val_loader, device, args.num_buckets)
        logger.info(
            "Epoch %d | Loss: %.4f | Acc: %.4f | Macro-F1: %.4f | Macro-P: %.4f | Macro-R: %.4f | Bal-Acc: %.4f",
            epoch + 1,
            avg_loss,
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            val_metrics["macro_precision"],
            val_metrics["macro_recall"],
            val_metrics["balanced_accuracy"],
        )

    # 6. 保存模型
    logger.info("Saving model...")
    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
    logger.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WorkloadProfiler on JSONL dataset.")
    parser.add_argument(
        "--dataset_path",
        default="/home/linchx/vidur/data/processed_traces/workload_profiler_train.jsonl",
        help="Path to JSONL dataset.",
    )
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--num_buckets", type=int, default=4)
    parser.add_argument("--num_task_types", type=int, default=3)
    parser.add_argument("--freeze_bert", action="store_true", default=True)
    parser.add_argument("--unfreeze_bert_layers", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", default="")
    args = parser.parse_args()

    train_model(args)

