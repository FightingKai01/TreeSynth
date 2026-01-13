import os

# Common configuration parameters
DEFAULT_CONFIG = {
    "max_depth": 6,#4
    "num_samples_per_node": 10,#10
    "max_attribute_count": 50,#50
    "max_sample_infinite_attribute": 25,#50
    "infinite_path_samples": 10,#10
    # "infinite_path_samples": 5,
    "max_workers": 64,#64
    "max_concurrent_requests": 64,#64
    "max_retries": 5,#5


    # --- 新增/修改参数 (针对 第一部分：深度控制) ---
    "mini_batch_size": 3,           # 小批次迭代的大小
    "saturation_threshold": 0.7,   # 相似度阈值 (>0.85 视为重复)
    "max_consecutive_saturation": 3,# 连续 K 次饱和则停止 (对应你提到的 K)
    "min_samples_per_node": 5,      # 节点最少生成多少个样本（防止太少无法分析）
    "max_samples_per_node": 25,     # 节点最多生成多少个样本（防止无限生成耗费 Token）

    # --- 新增/修改参数 (针对 第二部分：宽度/MCTS) ---
    "n_candidate_dimensions": 2,    # 每次让 LLM 提出几个候选维度进行 PK (MCTS 分支数)
    "simulation_batch_size": 10,     # 模拟阶段生成的样本数 (用于快速计算 Diversity)

    # [新增] 属性扩展相关配置
    "expand_mini_batch": 10,       # 每次请求让 LLM 生成多少个新属性
    "attr_similarity_threshold": 0.85, # 属性查重阈值 (高于此值视为重复)
    "max_expand_attempts_stuck": 3, # 如果连续 N 次生成的内容都被过滤掉了，强制停止


}

# # Backend-specific configurations
# BACKEND_CONFIGS = {
#     "vllm": {·
#         "api_base": "vllm-api-base",
#         "api_key": "vllm-api-key",
#         "model_name": "/path/to/model/qwen2_5-72b-instruct",
#     },
#     "azure": {
#         "api_key": "azure-api-key",
#         "model_name": "gpt-4o",
#         "azure_endpoint": "https://azure-endpoint.openai.azure.com/",
#         "api_version": "2024-10-21",
#     },
#     "openai": {
#         "api_key": "openai-api-key",
#         "model_name": "gpt-4o",
#     }
# }
BACKEND_CONFIGS = {
    "openai": {
        "api_base": "https://api.chatanywhere.tech/v1",
        "api_key": "sk-XmIRSMlKzuvKnKeOFlErHL9iHIKAjPSvCyEP92GxQuubRv0h",#chatanywhere
        "model_name": "gpt-4o",#https://openai.com/index/hello-gpt-4o 对应gpt-4o-2024-08-06
    }
}

# API pool configurations（可选：为了实现高吞吐量数据生成，请配置多个 API 端点）
VLLM_API_POOL = [
    {
        "api_base": "vllm-api-base",
        "api_key": "vllm-api-key",
        "model_name": "/path/to/model/qwen2_5-72b-instruct",
    },
]

OPENAI_API_POOL = [
    {
        "endpoint": "https://azure-endpoint.openai.azure.com/",
        "key": "azure-api-key1",
        "version": "2024-10-21",
        "model": "gpt-4o",
    },
    {
        "endpoint": "https://azure-endpoint.openai.azure.com/",
        "key": "azure-api-key2",
        "version": "2024-10-21",
        "model": "gpt-4o",
    }
]