import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional

class WorkloadProfilerConfig:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_buckets: int = 4,
        num_task_types: int = 3,
        embedding_dim: int = 32,
        prompt_len_dim: int = 16,
        hidden_dim: int = 128,
        dropout_prob: float = 0.1,
        freeze_bert: bool = True,
        unfreeze_bert_layers: int = 0,
        use_task_type: bool = False,
        use_prompt_len: bool = False,
    ):
        self.model_name = model_name
        self.num_buckets = num_buckets
        self.num_task_types = num_task_types
        self.embedding_dim = embedding_dim
        self.prompt_len_dim = prompt_len_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.freeze_bert = freeze_bert
        self.unfreeze_bert_layers = unfreeze_bert_layers
        self.use_task_type = use_task_type
        self.use_prompt_len = use_prompt_len

class WorkloadProfiler(nn.Module):
    """
    基于语义感知的异构工作负载分析器 (Semantics-Aware Heterogeneous Workload Profiler)
    
    架构:
    [Input 1: Prompt Text]       [Input 2: Task Metadata]    [Input 3: Prompt Len]
             |                              |
        (Tokenizer)                 (Label Encoding)
             |                              |
    +-------------------+        +----------------------+
    | DistilBERT Model  |        | Task Type Embedding  |
    | (Pre-trained)     |        | (Learnable Matrix)   |
    +-------------------+        +----------------------+
             |                              |
      [Text Feature Vector]      [Structure Feature Vector]
          (768-dim)                     (32-dim)
             |                              |
             +-------------+----------------+
                           |
                   [Concatenation]
                           |
                 +------------------+
                 |   MLP Head       |
                 | (ReLU + Dropout) |
                 +------------------+
                           |
                   [Softmax Output]
    """
    def __init__(self, config: WorkloadProfilerConfig):
        super().__init__()
        self.config = config
        
        # 1. 语义特征提取器 (Semantic Encoder)
        # 加载预训练模型配置和权重
        self.bert_config = AutoConfig.from_pretrained(config.model_name)
        self.bert = AutoModel.from_pretrained(config.model_name)
        
        # 可选：冻结 BERT 参数以减少计算量并防止过拟合
        if config.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            if config.unfreeze_bert_layers > 0:
                self._unfreeze_last_bert_layers(config.unfreeze_bert_layers)
        
        # 获取 BERT 输出维度 (通常是 768)
        self.text_hidden_size = self.bert.config.hidden_size
        
        # 2. 结构特征嵌入层 (Structural Embedding)
        # 简单的 Lookup Table，将任务类型 ID 映射为向量
        self.use_task_type = config.use_task_type
        if self.use_task_type:
            self.task_type_embedding = nn.Embedding(
                num_embeddings=config.num_task_types,
                embedding_dim=config.embedding_dim
            )

        # 2.5 Prompt length feature projection (scalar -> vector)
        self.use_prompt_len = config.use_prompt_len
        if self.use_prompt_len:
            self.prompt_len_proj = nn.Linear(1, config.prompt_len_dim)
        
        # 3. 多模态融合与推理 (Fusion & Inference)
        # 融合后的特征维度
        combined_dim = self.text_hidden_size
        if self.use_task_type:
            combined_dim += config.embedding_dim
        if self.use_prompt_len:
            combined_dim += config.prompt_len_dim
        
        # MLP 分类头
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.hidden_dim, config.num_buckets)
        )

    def _get_bert_layers(self):
        # Support common HF encoder layouts (BERT/DistilBERT/Roberta).
        if hasattr(self.bert, "encoder") and hasattr(self.bert.encoder, "layer"):
            return list(self.bert.encoder.layer)
        if hasattr(self.bert, "transformer") and hasattr(self.bert.transformer, "layer"):
            return list(self.bert.transformer.layer)
        return []

    def _unfreeze_last_bert_layers(self, num_layers: int) -> None:
        layers = self._get_bert_layers()
        if not layers:
            return
        num_layers = min(num_layers, len(layers))
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        task_type_ids: Optional[torch.Tensor] = None,
        prompt_len: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: BERT 输入 token IDs [batch_size, seq_len]
            attention_mask: BERT 注意力掩码 [batch_size, seq_len]
            task_type_ids: 任务类型 ID [batch_size] (例如: 0=Map, 1=Reduce, 2=ShareGPT)
            prompt_len: prompt 长度 (标量) [batch_size] 或 [batch_size, 1]
            
        Returns:
            logits: 分类 logits [batch_size, num_buckets]
        """
        # --- 1. 提取语义特征 ---
        # 输出包含 last_hidden_state: [batch_size, seq_len, hidden_size]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 取 [CLS] token 的向量作为句子表征 (通常是序列的第一个 token)
        # 第一个特殊 token 叫 [CLS]，它不是文本里真实的词，而是专门用来“汇总整句信息”的位置。模型训练时会把“整句的语义信息”聚到这个位置上
        text_features = outputs.last_hidden_state[:, 0, :] # [batch_size, text_hidden_size]
        
        # --- 2. 提取结构特征 ---
        features = [text_features]
        if self.use_task_type:
            if task_type_ids is None:
                raise ValueError("task_type_ids is required when use_task_type=True")
            task_features = self.task_type_embedding(task_type_ids) # [batch_size, embedding_dim]
            features.append(task_features)
        
        # --- 3. 特征融合 ---
        if self.use_prompt_len:
            if prompt_len is None:
                raise ValueError("prompt_len is required when use_prompt_len=True")
            if prompt_len.dim() == 1:
                prompt_len = prompt_len.unsqueeze(1)
            # 轻量归一化，避免过大数值主导
            prompt_len_norm = torch.log1p(prompt_len.float())
            prompt_features = self.prompt_len_proj(prompt_len_norm)
            features.append(prompt_features)

        combined_features = torch.cat(features, dim=1)
        
        # --- 4. 分类预测 ---
        logits = self.classifier(combined_features)
        
        return logits

    def predict(self, input_ids, attention_mask, task_type_ids=None, prompt_len=None):
        """用于推理的辅助方法，返回预测的 bucket 索引"""
        self.eval()
        with torch.no_grad():
            logits = self(input_ids, attention_mask, task_type_ids, prompt_len=prompt_len)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        return predictions, probs

# 简单的测试代码
if __name__ == "__main__":
    print("Initializing WorkloadProfiler...")
    config = WorkloadProfilerConfig()
    model = WorkloadProfiler(config)
    print(f"Model initialized. Structure:\n{model}")
    
    # 模拟输入数据
    batch_size = 2
    seq_len = 128
    
    # 随机生成文本输入
    dummy_input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    dummy_mask = torch.ones((batch_size, seq_len))
    
    # 随机生成任务类型 (0, 1, 2)
    dummy_task_types = torch.randint(0, 3, (batch_size,))
    dummy_prompt_len = torch.randint(1, 512, (batch_size,))
    
    print("\nRunning forward pass with dummy data...")
    logits = model(dummy_input_ids, dummy_mask, dummy_task_types, prompt_len=dummy_prompt_len)
    print(f"Logits shape: {logits.shape}")
    
    preds, probs = model.predict(dummy_input_ids, dummy_mask, dummy_task_types, prompt_len=dummy_prompt_len)
    print(f"Predictions: {preds}")
    print(f"Probabilities: {probs}")

