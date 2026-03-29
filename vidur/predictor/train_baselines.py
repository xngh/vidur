import argparse
"""
使用示例:
  - 随机森林: python -m vidur.predictor.train_baselines --model_type rf
  - TextCNN:  python -m vidur.predictor.train_baselines --model_type textcnn
  - BiLSTM:   python -m vidur.predictor.train_baselines --model_type bilstm
  - DistilBERT-only: python -m vidur.predictor.train_baselines --model_type distilbert_only
"""
import logging
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from vidur.predictor.train_profiler_example import (
    compute_metrics,
    load_jsonl,
    train_val_split,
)
from vidur.predictor.workload_profiler import WorkloadProfiler, WorkloadProfilerConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineTextDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item["text"]
        task_type = item["task_type"]
        label = item["label"]
        prompt_len = item.get("prompt_len")
        if prompt_len is None:
            prompt_len = len(str(text).split())

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "task_type_ids": torch.tensor(task_type, dtype=torch.long),
            "prompt_len": torch.tensor(prompt_len, dtype=torch.float),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class TextCNNBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        num_task_types: int,
        embedding_dim: int = 128,
        task_type_dim: int = 32,
        prompt_len_dim: int = 16,
        num_filters: int = 128,
        kernel_sizes: Tuple[int, ...] = (3, 4, 5),
        dropout_prob: float = 0.1,
        use_prompt_len: bool = True,
        use_task_type: bool = True,
    ):
        super().__init__()
        self.use_prompt_len = use_prompt_len
        self.use_task_type = use_task_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )
        self.task_type_embedding = nn.Embedding(num_task_types, task_type_dim)
        if self.use_prompt_len:
            self.prompt_len_proj = nn.Linear(1, prompt_len_dim)

        combined_dim = num_filters * len(kernel_sizes)
        if self.use_task_type:
            combined_dim += task_type_dim
        if self.use_prompt_len:
            combined_dim += prompt_len_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(combined_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_type_ids: torch.Tensor,
        prompt_len: torch.Tensor = None,
    ) -> torch.Tensor:
        emb = self.embedding(input_ids)  # [B, S, E]
        if attention_mask is not None:
            emb = emb * attention_mask.unsqueeze(-1)
        emb = emb.transpose(1, 2)  # [B, E, S]

        pooled = []
        for conv in self.convs:
            x = torch.relu(conv(emb))  # [B, F, S']
            x = torch.max(x, dim=2).values  # [B, F]
            pooled.append(x)
        text_features = torch.cat(pooled, dim=1)

        features = [text_features]
        if self.use_task_type:
            task_features = self.task_type_embedding(task_type_ids)
            features.append(task_features)

        if self.use_prompt_len:
            if prompt_len is None:
                raise ValueError("prompt_len is required when use_prompt_len=True")
            if prompt_len.dim() == 1:
                prompt_len = prompt_len.unsqueeze(1)
            prompt_len_norm = torch.log1p(prompt_len.float())
            prompt_features = self.prompt_len_proj(prompt_len_norm)
            features.append(prompt_features)

        combined = torch.cat(features, dim=1)
        return self.classifier(combined)


class BiLSTMBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        num_task_types: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        task_type_dim: int = 32,
        prompt_len_dim: int = 16,
        dropout_prob: float = 0.1,
        use_prompt_len: bool = True,
        use_task_type: bool = True,
    ):
        super().__init__()
        self.use_prompt_len = use_prompt_len
        self.use_task_type = use_task_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if num_layers > 1 else 0.0,
        )
        self.task_type_embedding = nn.Embedding(num_task_types, task_type_dim)
        if self.use_prompt_len:
            self.prompt_len_proj = nn.Linear(1, prompt_len_dim)

        combined_dim = hidden_dim * 2
        if self.use_task_type:
            combined_dim += task_type_dim
        if self.use_prompt_len:
            combined_dim += prompt_len_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(combined_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_type_ids: torch.Tensor,
        prompt_len: torch.Tensor = None,
    ) -> torch.Tensor:
        emb = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n shape: [num_layers * 2, B, hidden_dim]
        last_forward = h_n[-2]
        last_backward = h_n[-1]
        text_features = torch.cat([last_forward, last_backward], dim=1)

        features = [text_features]
        if self.use_task_type:
            task_features = self.task_type_embedding(task_type_ids)
            features.append(task_features)
        if self.use_prompt_len:
            if prompt_len is None:
                raise ValueError("prompt_len is required when use_prompt_len=True")
            if prompt_len.dim() == 1:
                prompt_len = prompt_len.unsqueeze(1)
            prompt_len_norm = torch.log1p(prompt_len.float())
            prompt_features = self.prompt_len_proj(prompt_len_norm)
            features.append(prompt_features)

        combined = torch.cat(features, dim=1)
        return self.classifier(combined)


def evaluate_torch_model(model, dataloader, device, num_classes: int) -> Dict[str, float]:
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


def train_torch_baseline(args) -> None:
    processed_data = load_jsonl(args.dataset_path)
    if not processed_data:
        raise ValueError(f"Dataset is empty: {args.dataset_path}")

    train_data, val_data = train_val_split(
        processed_data, val_ratio=args.val_ratio, seed=args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = BaselineTextDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = BaselineTextDataset(val_data, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    vocab_size = tokenizer.vocab_size
    if args.model_type == "textcnn":
        model = TextCNNBaseline(
            vocab_size=vocab_size,
            num_classes=args.num_buckets,
            num_task_types=args.num_task_types,
            embedding_dim=args.embedding_dim,
            task_type_dim=args.task_type_dim,
            prompt_len_dim=args.prompt_len_dim,
            num_filters=args.num_filters,
            kernel_sizes=tuple(args.kernel_sizes),
            dropout_prob=args.dropout_prob,
            use_prompt_len=args.use_prompt_len,
            use_task_type=args.use_task_type,
        )
    elif args.model_type == "bilstm":
        model = BiLSTMBaseline(
            vocab_size=vocab_size,
            num_classes=args.num_buckets,
            num_task_types=args.num_task_types,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            task_type_dim=args.task_type_dim,
            prompt_len_dim=args.prompt_len_dim,
            dropout_prob=args.dropout_prob,
            use_prompt_len=args.use_prompt_len,
            use_task_type=args.use_task_type,
        )
    else:
        raise ValueError(f"Unsupported model_type for torch: {args.model_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Using device: %s", device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            task_type_ids = batch["task_type_ids"].to(device)
            labels = batch["labels"].to(device)
            prompt_len = batch["prompt_len"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, task_type_ids, prompt_len=prompt_len)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        val_metrics = evaluate_torch_model(model, val_loader, device, args.num_buckets)
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


def train_distilbert_only_baseline(args) -> None:
    processed_data = load_jsonl(args.dataset_path)
    if not processed_data:
        raise ValueError(f"Dataset is empty: {args.dataset_path}")

    labels = [item.get("label") for item in processed_data]
    if any(label is None for label in labels):
        raise ValueError("Dataset has missing 'label' fields.")
    max_label = max(labels)
    if args.num_buckets <= max_label:
        args.num_buckets = max_label + 1
        logger.warning(
            "num_buckets is too small for labels. Auto-adjusting to %d.",
            args.num_buckets,
        )

    train_data, val_data = train_val_split(
        processed_data, val_ratio=args.val_ratio, seed=args.seed
    )

    config = WorkloadProfilerConfig(
        model_name=args.model_name,
        num_buckets=args.num_buckets,
        num_task_types=args.num_task_types,
        hidden_dim=args.hidden_dim,
        dropout_prob=args.dropout_prob,
        freeze_bert=args.freeze_bert,
        unfreeze_bert_layers=args.unfreeze_bert_layers,
        use_task_type=False,
        use_prompt_len=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = WorkloadProfiler(config)

    train_dataset = BaselineTextDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = BaselineTextDataset(val_data, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Using device: %s", device)
    logger.info("Training DistilBERT-only baseline without task metadata or prompt length.")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        val_metrics = evaluate_torch_model(model, val_loader, device, args.num_buckets)
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


def train_rf_baseline(args) -> None:
    processed_data = load_jsonl(args.dataset_path)
    if not processed_data:
        raise ValueError(f"Dataset is empty: {args.dataset_path}")

    texts = [item["text"] for item in processed_data]
    labels = [item["label"] for item in processed_data]
    task_types = [item["task_type"] for item in processed_data]
    prompt_lens = [
        item.get("prompt_len", len(str(item["text"]).split())) for item in processed_data
    ]

    text_train, text_val, y_train, y_val, t_train, t_val, p_train, p_val = train_test_split(
        texts,
        labels,
        task_types,
        prompt_lens,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        max_features=args.tfidf_max_features,
        ngram_range=tuple(args.tfidf_ngrams),
    )
    x_train_text = vectorizer.fit_transform(text_train)
    x_val_text = vectorizer.transform(text_val)

    svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.seed)
    x_train_reduced = svd.fit_transform(x_train_text)
    x_val_reduced = svd.transform(x_val_text)

    train_numeric = np.stack(
        [np.array(t_train, dtype=np.float32), np.log1p(np.array(p_train, dtype=np.float32))],
        axis=1,
    )
    val_numeric = np.stack(
        [np.array(t_val, dtype=np.float32), np.log1p(np.array(p_val, dtype=np.float32))],
        axis=1,
    )

    x_train = np.concatenate([x_train_reduced, train_numeric], axis=1)
    x_val = np.concatenate([x_val_reduced, val_numeric], axis=1)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_val)

    metrics = compute_metrics(list(preds), list(y_val), args.num_buckets)
    logger.info(
        "RF | Acc: %.4f | Macro-F1: %.4f | Macro-P: %.4f | Macro-R: %.4f | Bal-Acc: %.4f",
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["macro_precision"],
        metrics["macro_recall"],
        metrics["balanced_accuracy"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models for WorkloadProfiler.")
    parser.add_argument(
        "--dataset_path",
        default="/home/linchx/vidur/data/processed_traces/workload_profiler_train.jsonl",
    )
    parser.add_argument(
        "--model_type",
        choices=["rf", "textcnn", "bilstm", "distilbert_only"],
        default="rf",
    )
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--num_buckets", type=int, default=4)
    parser.add_argument("--num_task_types", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_bert", action="store_true", default=True)
    parser.add_argument("--unfreeze_bert_layers", type=int, default=0)

    # Torch model params
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--task_type_dim", type=int, default=32)
    parser.add_argument("--prompt_len_dim", type=int, default=16)
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--kernel_sizes", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--use_prompt_len", action="store_true", default=True)
    parser.add_argument("--use_task_type", action="store_true", default=True)

    # RF params
    parser.add_argument("--tfidf_max_features", type=int, default=20000)
    parser.add_argument("--tfidf_ngrams", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--svd_dim", type=int, default=256)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--n_jobs", type=int, default=-1)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model_type == "rf":
        train_rf_baseline(args)
    elif args.model_type == "distilbert_only":
        train_distilbert_only_baseline(args)
    else:
        train_torch_baseline(args)


if __name__ == "__main__":
    main()
