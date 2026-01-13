import json
import random
import logging
from collections import deque
from tqdm import tqdm
import os
import torch
import numpy as np
import sys
from openai import OpenAI, AzureOpenAI
from PrettyPrint import PrettyPrintTree
import re
import asyncio, threading
import httpx
import concurrent.futures
import time
from numpy.linalg import norm
from config import *


# ==========================================
# [新增] 1. 迭代采样器：负责深度控制与饱和度检测
# ==========================================
class IterativeSampler:
    """
        迭代采样器类 (IterativeSampler)
        功能：
            1. 负责在给定节点上下文中，分批次生成样本。
            2. 利用 Embedding 模型计算新旧样本的语义相似度。
            3. 根据相似度阈值判断当前空间是否饱和，从而决定是否停止生成。
    """
    def __init__(self, llm_engine, config):
        """
                初始化采样器
                :param llm_engine: (LLMInference) 主 LLM 推理引擎实例，用于调用生成接口和获取 logger
                :param config: (dict) 全局配置字典，包含 api_key, max_samples 等超参数
        """
        # 保存 LLM 引擎引用，后续用于调用 generate_batch_async
        self.llm_engine = llm_engine
        self.config = config
        # 复用引擎的 logger，保持日志格式统一
        self.logger = llm_engine.logger
        # 专门用于 Embedding 的客户端 (根据需求强制使用 OpenAI)
        # 注意：这里假设 config 中有 OpenAI 的 key，即使生成用的是 vLLM

        # todo embeddding模型的调用暂时使用chatanywhere API 支持的嵌入模型有(text-embedding-ada-002,text-embedding-3-small,text-embedding-3-large) 后续考虑使用部署到本地的embedding模型
        self.embedding_client = OpenAI(
            api_key=config.get("api_key"),
            base_url=config.get("api_base")
        )
        # 指定使用的 Embedding 模型，text-embedding-3-small 性价比高且性能足够
        self.embedding_model = "text-embedding-3-small"

    def get_embeddings(self, texts):
        """
                调用 OpenAI API 获取文本向量

                :param texts: (List[str]) 需要向量化的文本列表
                :return: (List[List[float]]) 返回二维浮点数列表，表示每个文本的 Embedding 向量
        """
        """调用 OpenAI API 获取文本向量"""
        if not texts:
            return []
        try:
            # 预处理：移除换行符以获得更好的 embedding 效果（OpenAI 官方建议）
            clean_texts = [t.replace("\n", " ") for t in texts]
            # 调用 OpenAI Embedding 接口
            response = self.embedding_client.embeddings.create(
                input=clean_texts,
                model=self.embedding_model,
            )
            # 提取并返回向量数据
            return [data.embedding for data in response.data]
        except Exception as e:
            # 异常处理：记录错误并返回空向量，防止程序崩溃
            self.logger.error(f"Embedding error: {e}")
            return [[] for _ in texts]  # 发生错误返回空向量


    async def get_embeddings_async(self, texts):
        """
        修改为异步

        """
        if not texts:
            return []
        clean_texts = [t.replace("\n", " ") for t in texts]
        max_retries = 5
        retries = 0
        backoff = 1.0

        while retries < max_retries:
            retries += 1
            try:
                # 使用 asyncio.to_thread 将同步的 client.create 放到线程池运行，防止阻塞
                response = await asyncio.to_thread(
                    self.embedding_client.embeddings.create,
                    input=clean_texts,
                    model=self.embedding_model,
                )
                return [data.embedding for data in response.data]
            except Exception as e:
                self.logger.error(f"Embedding error: {e}. Retry {retries} in {backoff}s...")
                # 非阻塞等待
                await asyncio.sleep(backoff)
                backoff *= 2

        self.logger.error("Embedding failed fully.")
        return [[] for _ in texts]

    def filter_new_samples(self, new_embeddings, history_embeddings, new_samples, threshold):
        """
        批量筛选样本
        功能：
            采用“滚动更新”的策略。维护一个对比池（初始为历史数据）。
            遍历新样本，每当确认一个有效样本，立刻加入对比池，供后续样本查重。
            这自然涵盖了：1.与历史查重 2.与当前批次内已通过的样本查重。

        :param new_embeddings: (List[List[float]]) 本次新生成样本的向量列表
        :param history_embeddings: (List[List[float]]) 历史已保存样本的向量列表
        :param new_samples: (List[str]) 本次新生成的文本列表
        :param threshold: (float) 判定重复的相似度阈值
        :return: (tuple)
            - valid_samples (List[str]): 有效样本列表
            - valid_embeddings (List[List[float]]): 有效样本向量列表
            - has_duplicate_in_batch (bool): 是否检测到重复（用于饱和度判定）
        """
        # 边界检查
        if not new_embeddings:
            return [], [], False

        valid_samples = []
        valid_embeddings = []
        has_duplicate_in_batch = False

        # [核心优化] 建立对比池，初始状态即为历史数据
        # 使用 list 复制一份，避免修改外部传入的引用，同时也方便动态 append
        comparison_pool = list(history_embeddings) if history_embeddings else []

        # 遍历当前批次的每一个样本
        for i, nv in enumerate(new_embeddings):
            nv_norm = norm(nv)
            if nv_norm == 0: continue

            is_duplicate = False

            # 只有对比池不为空时才计算（第一批的第一个样本会跳过此步直接视为有效）
            if comparison_pool:
                # 将列表转为矩阵进行广播计算 (N_pool, D)
                # 虽然每次循环都转换有开销，但考虑到 max_samples 规模(几十个)和 mini_batch(3-5个)，
                # 这里的性能损耗可以忽略，换取代码的极致简洁。
                pool_mat = np.array(comparison_pool)
                pool_norms = norm(pool_mat, axis=1)
                pool_norms[pool_norms == 0] = 1e-9

                # 计算当前向量 nv 与 池中所有向量的相似度
                sims = np.dot(pool_mat, nv) / (pool_norms * nv_norm)

                if np.max(sims) > threshold:
                    is_duplicate = True

            if is_duplicate:
                has_duplicate_in_batch = True
            else:
                # [核心逻辑] 样本有效：
                # 1. 放入返回列表
                valid_samples.append(new_samples[i])
                valid_embeddings.append(nv)
                # 2. 立即加入对比池！
                # 这样下一次循环时，它就成为了“历史”的一部分，后续样本会跟它进行比对
                comparison_pool.append(nv)

        return valid_samples, valid_embeddings, has_duplicate_in_batch

    async def generate_until_saturated(self, node_context_prompt, existing_samples=None):
        """
        核心迭代生成逻辑 (Iterative Generation)
        [修改说明]：
            1. 每次生成后，保留“未重复”的样本加入库中。
            2. 饱和判定逻辑变更为：如果连续 max_consecutive 个批次中，
               都出现了至少一个重复样本（即 LLM 开始频繁生成旧内容），则认为空间饱和。

        :param node_context_prompt: (str) 当前节点的生成 Prompt 基础模板
        :param existing_samples: (List[str] | None) 已有的样本


        :return: (tuple) (final_samples, is_saturated)
        """
        # 初始化样本库
        samples = existing_samples if existing_samples else []
        embeddings = await self.get_embeddings_async(samples) if samples else []

        saturation_counter = 0  # 连续出现重复样本的批次计数

        # 读取配置
        mini_batch = self.config.get("mini_batch_size", 3)
        max_samples = self.config.get("max_samples_per_node", 20)
        threshold = self.config.get("saturation_threshold", 0.85)
        max_consecutive = self.config.get("max_consecutive_saturation", 3)

        self.logger.info(f"Start iterative sampling. Init count: {len(samples)}")

        while len(samples) < max_samples:
            # 1. 构造“对抗式” Prompt
            # 将当前所有有效样本放入 Prompt，要求生成不同的 (注意 Token 长度限制，样本极多时需截断)
            current_prompt = self.llm_engine.format_iterative_gen_prompt(node_context_prompt, samples)

            # 2. 生成 Mini-Batch

            # todo 暂时不采用AI给的这种方式，会导致生成 9个结果，在这个中及逆行筛选
            # 我们请求生成 mini_batch 个，但为了防止 LLM 在一个 response 里偷懒，
            # 这里简单起见，调用 mini_batch 次并发请求，或者让 Prompt 一次吐出多个。
            # 这里复用 generate_batch_async 并发请求
            # batch_prompts = [current_prompt] * mini_batch

            responses = await self.llm_engine.generate_batch_async([current_prompt])# todo 关键点，将prompt转化成列表，视为多个任务，返回得到的就是 responses列表

            # 解析响应
            new_samples_raw = []
            pattern = r"Question\s+(\d+)\s*:\s*(.*?)(?=\s*Question\s+\d+:|$)"
            for raw_text in responses:#可有正确解析对应的问题
                matches = re.findall(pattern, raw_text, flags=re.DOTALL)
                for _, qtext in matches:
                    if qtext.strip():
                        new_samples_raw.append(qtext.strip())

            if not new_samples_raw:
                self.logger.warning("No valid samples generated in this batch.")
                break

            # 3. 获取向量并执行筛选
            new_embeddings_raw = await self.get_embeddings_async(new_samples_raw)

            # 调用新的筛选函数
            # valid_samples: 本次幸存的样本
            # has_duplicate: 本次是否产生了垃圾数据（重复数据）
            valid_samples, valid_embeddings, has_duplicate = self.filter_new_samples(
                new_embeddings_raw, embeddings, new_samples_raw, threshold
            )

            # 4. 更新状态与饱和度检测
            if valid_samples:
                # 只有当有有效样本时，才加入历史库
                samples.extend(valid_samples)
                embeddings.extend(valid_embeddings)
                self.logger.info(
                    f"Batch added {len(valid_samples)}/{len(new_samples_raw)} samples. Total: {len(samples)}")

            # [核心修改] 饱和计数逻辑
            if has_duplicate:
                # 只要这批里有至少一个重复的，就记一次过错
                saturation_counter += 1
                self.logger.info(f"Duplicate detected in batch ({saturation_counter}/{max_consecutive}).")
            else:
                # 如果这批全是新的，说明空间还很广阔，重置计数器
                saturation_counter = 0

            # 5. 终止条件判定
            # 连续 K 次生成的批次里都包含重复内容，说明 LLM 已经很难想出完全不一样的了
            if saturation_counter >= max_consecutive:
                self.logger.info(f"Space saturated after {len(samples)} samples (consecutive duplicates).")
                return samples, True

        self.logger.info(f"Sampling stopped. Count: {len(samples)} (Max limit reached).")
        return samples, False


# ==========================================
# [新增] 2. MCTS 决策器：负责宽度控制与维度选择
# ==========================================
import numpy as np
from numpy.linalg import norm



class MCTSDecisionMaker:
    def __init__(self, sampler, logger):
        self.sampler = sampler
        self.logger = logger
        self.sim_batch_size = sampler.config.get("simulation_batch_size", 5)

    async def decide_best_split(self, parent_node, candidate_node_groups):
        """
        [MCTS 核心逻辑 - 预构建版]
        输入不再是 JSON，而是已经实例化好的 TreeNode 列表分组。
        结构：[ [GroupA_Child1, GroupA_Child2], [GroupB_Aggregate_Child] ]
        """
        scores = []

        self.logger.info(f"[MCTS] Evaluating {len(candidate_node_groups)} prepared candidate groups...")

        for idx, children_group in enumerate(candidate_node_groups):
            if not children_group:
                scores.append(float('inf'))
                continue

            # 获取维度名称 (取第一个孩子的维度名即可)
            dim_name = children_group[0].dimension

            # [优化策略]
            # 如果是 Finite 分组且孩子特别多（例如 20 个），为了节省 Token，
            # 模拟阶段可以只随机抽样 3-5 个孩子进行生成打分。
            # 如果是 Infinite 分组（只有 1 个孩子），则必须跑这一个。

            nodes_to_simulate = children_group# 是treenode的列表

            if len(children_group) > 5:# todo 删除随机模拟的操作，增加判断是否是无限节点的逻辑
                # 随机抽样模拟，避免全量跑
                import random
                nodes_to_simulate = random.sample(children_group, 5)
                self.logger.info(
                    f"  > Dim '{dim_name}': Too many children ({len(children_group)}). Sampling 5 for simulation.")
            else:
                self.logger.info(f"  > Dim '{dim_name}': Simulating all {len(children_group)} children.")

            total_sim_score = 0
            valid_sim_count = 0

            for child in nodes_to_simulate:
                # 1. [Simulation] 真实生成
                # 复用 generate_samples_async，自动处理饱和度
                samples, is_saturated = await child.generate_samples_async()

                # 2. [Scoring]
                sim_score = 1.0  # 默认最差

                if len(samples) > 1:
                    embs = await self.sampler.get_embeddings_async(samples)
                    if embs:
                        mat = np.array(embs)
                        norm_mat = norm(mat, axis=1, keepdims=True)
                        norm_mat[norm_mat == 0] = 1e-9
                        mat_normalized = mat / norm_mat
                        sim_matrix = np.dot(mat_normalized, mat_normalized.T)
                        np.fill_diagonal(sim_matrix, 0)
                        avg_sim = np.sum(sim_matrix) / (len(samples) * (len(samples) - 1))
                        sim_score = avg_sim

                # 记录单个节点得分
                child.diversity_score = sim_score
                total_sim_score += sim_score
                valid_sim_count += 1

            # 计算维度平均分
            if valid_sim_count > 0:
                avg_dim_score = total_sim_score / valid_sim_count
            else:
                avg_dim_score = 1.0

            scores.append(avg_dim_score)
            self.logger.info(f"  > Dim '{dim_name}' Avg Score: {avg_dim_score:.4f}")

        # [Selection]
        best_idx = np.argmin(scores)
        best_dim_name = candidate_node_groups[best_idx][0].dimension
        self.logger.info(f"[MCTS] Winner: '{best_dim_name}' (Score: {scores[best_idx]:.4f})")

        # [State Update]
        active_children = []
        for idx, children_group in enumerate(candidate_node_groups):
            is_winner = (idx == best_idx)
            for child in children_group:
                child.is_selected = is_winner
                if is_winner:
                    active_children.append(child)

        return active_children


# ==========================================
# 原有类定义 (部分修改)
# ==========================================

class APIPool:
    """Generic API pool that works for both vLLM and OpenAI backends"""

    def __init__(self, api_configs, requests_per_minute=1000):
        self.configs = api_configs
        self.index = 0
        self.lock = threading.Lock()
        self.error_counts = {i: 0 for i in range(len(api_configs))}
        self.requests_per_minute = requests_per_minute
        self.min_delay = 60.0 / requests_per_minute
        self.last_request_time = {i: 0 for i in range(len(self.configs))}

    def get_next_config(self):
        with self.lock:
            best_index = self.index
            min_errors = float('inf')
            for i in range(len(self.configs)):
                idx = (self.index + i) % len(self.configs)
                if self.error_counts[idx] < min_errors:
                    min_errors = self.error_counts[idx]
                    best_index = idx
            current_time = time.time()
            time_since_last = current_time - self.last_request_time[best_index]
            if time_since_last < self.min_delay:
                time.sleep(self.min_delay - time_since_last)
            self.index = (best_index + 1) % len(self.configs)
            self.last_request_time[best_index] = time.time()
            return self.configs[best_index], best_index

    def report_error(self, index):
        with self.lock:
            self.error_counts[index] += 1

    def report_success(self, index):
        with self.lock:
            self.error_counts[index] = 0


class LLMInference:
    def __init__(self, backend="vllm", api_pool=None, config=None, logger=None, max_retries=5, max_workers=20,
                 max_concurrent_requests=64):
        self.logger = logger or logging.getLogger()
        self.backend = backend.lower().strip()
        self.max_retries = max_retries
        self.threadpool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.api_pool = api_pool
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.config = config or {}

        # 初始化 Clients (保持不变)
        if self.backend == "vllm":
            self.model_name = self.config.get("model_name")
            if api_pool is None:
                self.client = OpenAI(base_url=self.config.get("api_base"), api_key=self.config.get("api_key"))
        elif self.backend == "azure":
            self.model_name = self.config.get("model_name", "gpt-4o")
            if api_pool is None:
                self.client = AzureOpenAI(azure_endpoint=self.config.get("azure_endpoint"),
                                          api_key=self.config.get("api_key"),
                                          api_version=self.config.get("api_version"))
        elif self.backend == "openai":
            self.model_name = self.config.get("model_name", "gpt-4o")
            if api_pool is None:
                self.client = OpenAI(base_url=self.config.get("api_base"), api_key=self.config.get("api_key"))
        else:
            raise ValueError(f"backend must be 'vllm' or 'openai', got {self.backend}.")

        # [新增] 初始化 Sampler 和 DecisionMaker
        self.sampler = IterativeSampler(self, self.config)
        self.decision_maker = MCTSDecisionMaker(self.sampler, self.logger)

    async def generate_per_prompt_async(self, prompt, max_tokens=1024, temperature=0.7):
        # ... (保持原有的 API 调用逻辑不变，包括 Pool 处理和重试逻辑) ...
        # 为节省篇幅，此处省略具体的 HTTP 调用代码，与原文件保持一致
        # 请确保保留原有的 generate_per_prompt_async 实现
        assert isinstance(prompt, str), "Prompt must be a string."
        retries = 0
        ans = ""
        backoff = 1.0
        while retries < self.max_retries:
            retries += 1
            async with self.semaphore:
                try:
                    if self.api_pool:
                        if self.backend == "vllm":
                            conf, conf_idx = self.api_pool.get_next_config()
                            local_client = OpenAI(base_url=conf["api_base"], api_key=conf["api_key"])
                            model_name = conf["model_name"]
                        elif self.backend == "azure":
                            conf, conf_idx = self.api_pool.get_next_config()
                            local_client = AzureOpenAI(azure_endpoint=conf["endpoint"], api_key=conf["key"],
                                                       api_version=conf["version"])
                            model_name = conf["model"]
                        elif self.backend == "openai":
                            conf, conf_idx = self.api_pool.get_next_config()
                            local_client = OpenAI(base_url=conf["api_base"], api_key=conf["api_key"])
                            model_name = conf["model_name"]
                    else:
                        local_client = self.client
                        model_name = self.model_name

                    completion = await asyncio.to_thread(
                        local_client.chat.completions.create,
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    ans = completion.choices[0].message.content.strip()
                    if self.api_pool: self.api_pool.report_success(conf_idx)
                    break
                except Exception as e:
                    if self.api_pool: self.api_pool.report_error(conf_idx)
                    self.logger.error(f"Error: {e}")
                    await asyncio.sleep(backoff)
                    backoff *= 2
        return ans

    async def generate_batch_async(self, prompts, max_tokens=1024, temperature=0.7):
        if isinstance(prompts, str):#当只有一个提示词，并非提示词列表时 只创建一个任务,但是返回的reponses为单一的字符串，并非列表
            return await self.generate_per_prompt_async(prompts, max_tokens, temperature)
        tasks = [asyncio.create_task(self.generate_per_prompt_async(p, max_tokens, temperature)) for p in prompts]
        return await asyncio.gather(*tasks)

    # ==========================================
    # [修改] Prompt 模板：支持迭代生成
    # ==========================================
    def format_iterative_gen_prompt(self, base_prompt, history_samples):
        """
        在基础 Prompt 上注入历史样本，构造对抗式 Prompt。
        动态计算接下来生成的题目编号 (start_index) 和 数量 (mini_batch_size)。
        """
        if not history_samples:
            return base_prompt

        # 1. 获取动态配置
        batch_size = self.config.get("mini_batch_size", 3)

        # 2. 计算编号接续逻辑
        existing_count = len(history_samples)
        start_index = existing_count + 1
        end_index = existing_count + batch_size

        # 3. 格式化历史样本
        history_text = "\n".join([f"- {s}" for s in history_samples])

        # 4. 构造指令
        # 注意：这里明确指定了 start_index 和 end_index，强制 LLM 续写编号
        iterative_instruction = f"""
        \n\nIMPORTANT: I have already generated the above {existing_count} questions. 
        You MUST generate {batch_size} NEW questions that are significantly DIFFERENT from these in terms of scenario, numbers, or logic.
        DO NOT REPEAT existing patterns.

        Existing Questions:
        {history_text}

        New Questions Requirement:
        1. Generate exactly {batch_size} new questions.
        2. Start numbering from Question {start_index} and end at Question {end_index}.

        Output strictly in the following format:
        Question {start_index}: text
        ...
        Question {end_index}: text
        """

        """
        \n\n重要提示：我已经生成了上述 {existing_count} 个问题。
        你务必生成 {batch_size} 个新问题，且必须在场景、数值或逻辑上与这些问题存在显著差异。
        切勿重复现有的模式。
        
        现有问题：
        {history_text}
        
        新问题要求：
        1. 准确生成 {batch_size} 个新问题。
        2. 编号从 Question {start_index} 开始，到 Question {end_index} 结束。
        
        严格按照以下格式输出：
        Question {start_index}: 文本
        ...
        Question {end_index}: 文本
        """

        # 将新指令追加到原 prompt 之后
        # LLM 通常会遵循最后出现的指令（Recency Bias），从而覆盖 Base Prompt 里的 "Question 1" 指令
        return base_prompt + iterative_instruction

    # ==========================================
    # [修改] Prompt 模板：支持 MCTS 维度选择
    # ==========================================
    def format_mcts_dim_prompt(self, samples, dimensions, n_candidates=2):
        """ todo 弃用  请求 LLM 生成多个候选维度，采用 循环调用select_dimension_and_classify_async生成候选维度和对应分类属性，并增加一个temp_excluded_dims用于在生成第二个、第三个候选时，排除掉前面已经生成的候选"""
        samples_str = "\n".join([f"{i}. {s}" for i, s in enumerate(samples, 1)])

        prompt = f"""
        As an analysis expert, your task is to analyze the following math word problems and propose {n_candidates} DISTINCT and ORTHOGONAL candidate dimensions to structure this data space.

        Questions:
        {samples_str}

        Dimension Requirements:
        1. Core Dimension Identification: Identify exactly ONE core dimension that best distinguishes these questions.
        2. Excluded Dimensions: {', '.join(dimensions)}
        3. Unique Categorization: Each question MUST be categorized into exactly ONE attribute value.
        4. Mutually Exclusive Values: Attribute values must be mutually exclusive.
        5. Clarity in Values: Avoid ambiguous attribute values, such as "others".
        6. Independent Values: Each attribute must be a single distinct value - NO combined values like "attribute1_and_attribute2" or "attribute1/attribute2"! Each attribute must be a single distinct value - NO combined values like "attribute1_and_attribute2" or "attribute1/attribute2"! Each attribute must be a single distinct value - NO combined values like "attribute1_and_attribute2" or "attribute1/attribute2"!
        

        Organize your responses in the following format without any extra text or explanations:
        {{
            "dimension": "Candidate Dimension 1",
            "attributes": {{ 
                "attribute1": [list of sample indices], 
                "attribute2": [list of sample indices] 
                }},
            ...
        }},
        {{
            "dimension": "Candidate Dimension 2",
            "attributes": {{ 
                "attribute1": [list of sample indices], 
                "attribute2": [list of sample indices] 
                }},
            ...
        }}
        """

        """
        作为数学分析专家，请识别出 {n_candidates} 个**独特**且**有效**的候选维度来对这些问题进行分类。

        问题列表：
        {samples_str}
        
        排除的维度：{', '.join(dimensions)}
        
        要求：
        1. 识别出 {n_candidates} 个不同的维度。
        2. 对于每个维度，将问题归类到互斥的属性中。
        3. 属性必须清晰、简单（初中水平），并覆盖所有样本。
        
        输出格式（JSON 列表）：
        [
            {{
                "dimension": "候选维度 1",
                "attributes": {{ "属性A": [索引列表], "属性B": [索引列表] }}
            }},
            {{
                "dimension": "候选维度 2",
                "attributes": {{ "属性C": [索引列表], "属性D": [索引列表] }}
            }}
        ]
        
        """

        return prompt

    def format_gen_prompt_for_simulation(self, node, dimension, attribute):
        """为 MCTS 模拟阶段构造临时的生成 Prompt"""
        # 类似 TreeNode.format_gen_prompt，但只需要针对特定 dim/attr
        # 这是一个简化版，复用 retrieve_dimension_values 逻辑比较复杂，这里手动构造
        parent_ctx = node.retrieve_dimension_values()
        # 加入当前模拟的 dim/attr
        parent_ctx.append({"dimension": dimension, "attribute_value": attribute})
        attr_json = json.dumps(parent_ctx, indent=2, ensure_ascii=False)

        prompt = f"""
        Generate {self.config.get("simulation_batch_size", 10)} GSM8K-style math word problems.
        Target Audience: Middle School.
        Constraints: {attr_json}
        Format:
        Question 1: text
        ...
        Question {self.config.get("simulation_batch_size", 10)}: text
        """
        return prompt

    # ... (保留原有的 format_gen_prompt, format_expand_prompt 等，用于常规流程) ...
    # 为了兼容旧代码调用，这里需要把 generator_gsm_async.py 中 TreeNode 类里的 format 方法
    # 逻辑保留在 TreeNode 中，或者搬运到这里。
    # *建议*：保持 TreeNode 中的 prompt 逻辑，只在 LLMInference 中添加新逻辑。


# ==========================================
# [修改] TreeNode 类：增加 MCTS 属性
# ==========================================
class TreeNode:
    def __init__(self, depth, llm_engine=None, parent=None, dimension=None, attribute_value=None,
                 max_depth=5, num_samples_per_node=10, infinite_threshold=10, max_attribute_count=20,
                 threadpool_executor=None, tree_structure_file="tree_structure.txt",
                 # [新增]
                 is_selected=True, diversity_score=0.0):

        # ... (原有初始化代码保持不变) ...
        if parent is None:
            assert depth == 0
        else:
            assert depth == parent.depth + 1
        self.depth = depth
        self.dimension = dimension
        self.attribute_value = attribute_value
        self.parent = parent
        self.children = []
        self.samples = None

        self.max_depth = max_depth
        self.num_samples_per_node = num_samples_per_node
        self.max_attribute_count = max_attribute_count
        self.infinite_threshold = infinite_threshold
        self.tree_structure_file = tree_structure_file

        self.llm_engine = llm_engine
        self.threadpool_executor = threadpool_executor
        self.logger = getattr(self.llm_engine, "logger", None)

        # [新增] MCTS 状态
        self.is_selected = is_selected  # 是否被 MCTS 选中为最优路径
        self.diversity_score = diversity_score  # 该节点的多样性得分 (越低越好)

    # ... (保留原有方法: is_leaf, is_root, is_infinite, __str__, logging) ...
    def logging(self, msg, level="info"):
        if self.logger: getattr(self.logger, level)(msg)

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def is_infinite(self, attribute_values):
        return len(attribute_values) > self.infinite_threshold


    def to_dict(self):
        """
        [序列化] 增加 MCTS 相关状态，方便在 output.json 中查看被舍弃节点的得分
        """
        return {
            "attribute_value": self.attribute_value,
            "dimension": self.dimension,
            "samples": self.samples,
            "is_selected": self.is_selected,  # 是否被选中
            "diversity_score": self.diversity_score,  # 多样性得分 (Debug关键)
            "children": [child.to_dict() for child in self.children],
        }
    def count_infinite_nodes_in_path(self):
        count = 0
        current = self
        while current:
            if isinstance(current.attribute_value, (list, set)):
                count += 1
            current = current.parent
        return count
    # ... (保留 retrieve_parents, retrieve_dimension_values, retrieve_parent_dimensions 等) ...
    def retrieve_parents(self):
        parents = []
        current = self
        while current:
            parents.append(current)
            current = current.parent
        return parents

    def retrieve_dimension_values(self):
        parents = self.retrieve_parents()
        parents = parents[:-1]
        parents.reverse()
        dimensions = []
        for parent in parents:
            dim = parent.dimension
            value = parent.attribute_value
            if isinstance(value, (list, set)):#处理无限节点的 属性选择
                value = random.choice(list(value))
            dimensions.append({"dimension": dim, "attribute_value": value})
        return dimensions

    def retrieve_parent_dimensions(self):
        return [d["dimension"] for d in self.retrieve_dimension_values()]

    def retrieve_root(self):
        current = self
        while current.parent:
            current = current.parent
        return current

    def save_tree_structure(self, output_file):
        """
        [可视化] 在树结构图中显示 is_selected 和 diversity_score
        """
        root = self.retrieve_root()

        # 修改打印格式，增加 score 显示
        pt = PrettyPrintTree(
            lambda x: x.children,
            lambda
                x: f"dim: {x.dimension}\nattr: {x.attribute_value}\nsel: {x.is_selected}\nscore: {x.diversity_score:.2f}",
            orientation=PrettyPrintTree.Horizontal
        )
        tree_as_str = pt(root, return_instead_of_print=True)
        ansi_escape = re.compile(r"(?:\x1B[@-_][0-?]*[ -/]*[@-~])")
        tree_as_str = ansi_escape.sub("", tree_as_str)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(tree_as_str)

        self.logging(f"Tree structure saved to {output_file}.", level="info")
    # [保留] Prompt 生成函数 (format_dim_prompt, format_expand_prompt, format_gen_prompt)
    # 此处省略具体实现，请保持原代码中的这几个函数不变，
    # 唯一需要注意的是 format_gen_prompt 需要被 Sampler 调用，所以 Sampler 里的 logic 会依赖它

    def format_gen_prompt(self):

        if self.is_root():

            prompt = f"""
            As a math expert, you are tasked to generate {self.llm_engine.config.get("mini_batch_size", 3)} GSM8K-style math word problems suitable for a bright middle school student.

            Each question should meet the following criteria:
            1. Format: Write problems as real-world word problems that require mathematical reasoning to solve.
            2. Step Count: Require between 2 and 8 steps to solve.
            3. Operations: Utilize basic arithmetic operations: addition (+), subtraction (-), multiplication (*), and division (/).
            4. Complexity: Vary in context and complexity, but REMAIN ACCESSIBLE TO MIDDLE SCHOOL STUDENTS!
            5. Clarity: Provide clear, concise questions that encourage step-by-step calculations to reach the final answer.
            6. Language: Use natural, conversational language to describe situations while keeping problems clear and unambiguous.
            7. Diversity: Ensure that the questions are diverse and distinct from one another from all potential perspectives.

            Organize your responses in the following format without any extra text or explanations:
            Question 1: text
            Question 2: text
            ...
            Question {self.llm_engine.config.get("mini_batch_size", 3)}: text
            """

            """
            作为数学专家，你的任务是生成 10 道适合聪明初中生的 GSM8K 风格的数学应用题。

            每个问题应符合以下标准：
            1. 格式：将问题编写为需要数学推理才能解决的现实世界应用题。
            2. 步骤数量：解决问题需要 2 到 8 个步骤。
            3. 运算：利用基本算术运算：加 (+)、减 (-)、乘 (*)、除 (/)。
            4. 复杂性：在情境和复杂性上要多样化，但必须保持初中生能够理解！
            5. 清晰度：提供清晰、简明的问题，鼓励通过一步步的计算得出最终答案。
            6. 语言：使用自然、对话式的语言描述情境，同时保持问题清晰无歧义。
            7. 多样性：确保问题在所有潜在角度上都是多样且彼此独特的。

            请按照以下格式组织你的回答，不要包含任何额外的文本或解释：
            Question 1: 文本
            Question 2: 文本
            ...
            """

        else:
            attributes = self.retrieve_dimension_values()
            attributes_json = json.dumps(attributes, indent=2, ensure_ascii=False)
            prompt = f"""
            As a math expert, you are tasked to generate {self.llm_engine.config.get("mini_batch_size", 3)} GSM8K-style math word problems suitable for a bright middle school student.

            Each question should meet the following criteria:
            1. Format: Write problems as real-world word problems that require mathematical reasoning to solve.
            2. Step Count: Require between 2 and 8 steps to solve.
            3. Operations: Utilize basic arithmetic operations: addition (+), subtraction (-), multiplication (*), and division (/).
            4. Complexity: Vary in context and complexity, but REMAIN ACCESSIBLE TO MIDDLE SCHOOL STUDENTS!
            5. Clarity: Provide clear, concise questions that encourage step-by-step calculations to reach the final answer.
            6. Language: Use natural, conversational language to describe situations while keeping problems clear and unambiguous.
            7. Diversity: Ensure that the questions are diverse and distinct from one another from all potential perspectives.
            8. Attributes: Each problem should be associated with all these attributes: {attributes_json}

            Organize your responses in the following format without any extra text or explanations:
            Question 1: text
            Question 2: text
            ...
            Question {self.llm_engine.config.get("mini_batch_size", 3)}: text
            """
            """
            作为数学专家，你的任务是生成 10 道适合聪明初中生的 GSM8K 风格的数学应用题。

            每个问题应符合以下标准：
            1. 格式：将问题编写为需要数学推理才能解决的现实世界应用题。
            2. 步骤数量：解决问题需要 2 到 8 个步骤。
            3. 运算：利用基本算术运算：加 (+)、减 (-)、乘 (*)、除 (/)。
            4. 复杂性：在情境和复杂性上要多样化，但必须保持初中生能够理解！
            5. 清晰度：提供清晰、简明的问题，鼓励通过一步步的计算得出最终答案。
            6. 语言：使用自然、对话式的语言描述情境，同时保持问题清晰无歧义。
            7. 多样性：确保问题在所有潜在角度上都是多样且彼此独特的。
            8. 属性：每个问题应与所有这些属性相关联：{attributes_json}

            请按照以下格式组织你的回答，不要包含任何额外的文本或解释：
            Question 1: 文本
            Question 2: 文本
            ...
            Question 10: 文本
            """

        return prompt

    def format_dim_prompt(self, override_excluded=None):
        """generate prompt for selecting dimension and classifying"""

        assert self.samples is not None, "Samples must be generated first."
        samples = ""
        for i, s in enumerate(self.samples, 1):
            samples += f"""{i}. {s}\n"""
        samples = samples.strip()

        # todo 修改 dimensions = self.retrieve_parent_dimensions()
        # 确定排除列表
        if override_excluded is not None:
            dimensions = override_excluded# 已经包含retrieve_parent_dimensions
        else:
            dimensions = self.retrieve_parent_dimensions()

        prompt = f"""
        As an analysis expert, your task is to examine the following questions to identify the SINGLE most significant dimension that characterizes the question space and differentiates these questions.
        Questions:
        {samples}

        Dimension Requirements:
        1. Core Dimension Identification: Identify exactly ONE core dimension that best distinguishes these questions.
        2. Excluded Dimensions: {', '.join(dimensions)}
        3. Unique Categorization: Each question MUST be categorized into exactly ONE attribute value.
        4. Mutually Exclusive Values: Attribute values must be mutually exclusive.
        5. Clarity in Values: Avoid ambiguous attribute values, such as "others".
        6. Independent Values: Each attribute must be a single distinct value - NO combined values like "attribute1_and_attribute2" or "attribute1/attribute2"! 

        Organize your responses in the following format without any extra text or explanations:
        {{
        "dimension": "dimension_name",
        "attributes": {{
            "attribute1": [list of sample indices],
            "attribute2": [list of sample indices],
            ...
        }}
        }}
        Each attribute must be a single distinct value - NO combined values like "attribute1_and_attribute2" or "attribute1/attribute2"! Each attribute must be a single distinct value - NO combined values like "attribute1_and_attribute2" or "attribute1/attribute2"!
        **STRICTLY FORBIDDEN FORMAT (Do NOT do this):**
        {{
        "dimension": "dimension_name",
        "attributes": {{
            "attribute1_and_attribute2": [indices],   <-- ERROR: NEVER combine values with "and", "_", or "/"
            "attribute3/attribute4": [indices]        <-- ERROR: Ensure that every attribute is independent, even if attribute1 and attribute2 are semantically similar.
        }}
        }}
        Ensure every attribute is independent (e.g., use "attribute1" and "attribute2" as separate keys).
        
        
        """
        """
        作为分析专家，你的任务是检查以下问题，并识别出表征这些问题空间并区分这些问题的**唯一**最显著维度。
        问题列表：
        {samples}

        维度要求：
        1. 核心维度识别：准确识别出一个最能区分这些问题的核心维度。
        2. 排除的维度：{', '.join(dimensions)}
        3. 唯一分类：每个问题必须被归类到准确的一个属性值中。
        4. 互斥值：属性值必须是互斥的。
        5. 数值清晰：避免使用模棱两可的属性值，例如 "others"（其他）。
        6. 独立值：每个属性必须是一个单一且独特的数值——绝不能包含组合值，如 "attribute1_and_attribute2" 或 "attribute1/attribute2"！

        请按照以下格式组织你的回答，不要有任何额外的文本或解释：
        {
        "dimension": "维度名称",
        "attributes": {
            "属性1": [样本索引列表],
            "属性2": [样本索引列表],
            ...
            }
        }
        每个属性必须是一个单一且独特的数值——绝不能包含组合值，如 "attribute1_and_attribute2" 或 "attribute1/attribute2"！每个属性必须是一个单一且独特的数值——绝不能包含组合值，如 "attribute1_and_attribute2" 或 "attribute1/attribute2"！
        """

        # todo 自己增加了禁止输出的相关格式的案例
        return prompt

    def format_expand_prompt(self, dimension, attribute_values):
        # 获取动态配置的 batch size
        batch_size = self.llm_engine.config.get("expand_mini_batch", 10)
        prompt = f"""
        As an analysis expert, your task is to supplement the potential attribute values for a specified dimension in order to comprehensively model the entire space of questions.

        Dimension: {dimension}
        Exiting attributes values: {json.dumps(attribute_values, indent=2)}

        Requirements for New Attribute Values:
        1. Clarity: Avoid ambiguous values, such as "others".
        2. Mutual Exclusivity: Ensure that attribute values do not overlap. **STRICTLY NO DUPLICATES**: Do NOT generate values that are semantically similar to the "Existing attributes values". (e.g., if "Baking" exists, do NOT generate "Cooking")!Ensure that attribute values do not overlap. **STRICTLY NO DUPLICATES**: Do NOT generate values that are semantically similar to the "Existing attributes values". (e.g., if "Baking" exists, do NOT generate "Cooking")!Ensure that attribute values do not overlap. **STRICTLY NO DUPLICATES**: Do NOT generate values that are semantically similar to the "Existing attributes values". (e.g., if "Baking" exists, do NOT generate "Cooking")!
        3. Completeness: Ensure that all possible attribute values fully cover the dimension.
        4. GRADE LEVEL: Keep all values within elementary and middle school students' understanding! Keep all values within elementary and middle school students' understanding! Keep all values within elementary and middle school students' understanding! 
        5. SIMPLICITY: Use basic, straightforward terms that young students can understand! Use basic, straightforward terms that young students can understand! Use basic, straightforward terms that young students can understand! 

        Organize your responses in the following format without any extra text or explanations:
        - If the existing attribute values completely cover the entire dimension, only output "null". For example,
        null
        - If the number of potential attribute values is more than 10, first output 10 potential new attribute values, and end your output with "infinite" in a new line. For example,
        attribute value 1
        attribute value 2
        ...
        attribute value {batch_size}
        infinite
        - Otherwise, output all the potential new attribute values, and end your output with "complete" in a new line. For example,
        attribute value 1
        attribute value 2
        ...
        attribute value n
        complete
        
        **STRICTLY FORBIDDEN FORMAT (Do NOT do this):**
        {{
        "dimension": "dimension_name",
        "attributes": {{
            - attribute value 1,   <-- ERROR: Adding any sequence-related typographical symbols, such as "-", "·", etc., before the attribute value is prohibited.
            · attribute value 2,   <-- ERROR: Adding any sequence-related typographical symbols, such as "-", "·", etc., before the attribute value is prohibited.
            ...
        }}
        }}

        """
        """
        作为分析专家，你的任务是为指定维度补充潜在的属性值，以便全面地对整个问题空间进行建模。

        维度：{dimension}
        现有属性值：{json.dumps(attribute_values, indent=2)}

        新属性值的要求：
        1. 清晰度：避免使用模棱两可的值，例如 "others"（其他）。
        2. 互斥性：确保属性值互不重叠。
        3. 完整性：确保所有可能的属性值完全覆盖该维度。**严格禁止重复**：请勿生成与“现有属性值”语义相似的值。（例如，如果“烘焙”已存在，请勿生成“烹饪”）!确保所有可能的属性值完全覆盖该维度。**严格禁止重复**：请勿生成与“现有属性值”语义相似的值。（例如，如果“烘焙”已存在，请勿生成“烹饪”）!确保所有可能的属性值完全覆盖该维度。**严格禁止重复**：请勿生成与“现有属性值”语义相似的值。（例如，如果“烘焙”已存在，请勿生成“烹饪”）!
        4. 年级水平：所有值必须在小学和初中学生理解范围内！所有值必须在小学和初中学生理解范围内！所有值必须在小学和初中学生理解范围内！
        5. 简单性：使用年轻学生能听懂的基本、直白的术语！使用年轻学生能听懂的基本、直白的术语！使用年轻学生能听懂的基本、直白的术语！

        请按照以下格式组织你的回答，不要包含任何额外的文本或解释：
        - 如果现有属性值已完全覆盖整个维度，仅输出 "null"。例如：
        null
        - 如果潜在属性值的数量超过 10 个，先输出 10 个潜在的新属性值，并在新的一行以 "infinite" 结束输出。例如：
        属性值 1
        属性值 2
        ...
        属性值 10
        infinite
        - 否则，输出所有潜在的新属性值，并在新的一行以 "complete" 结束输出。例如：
        属性值 1
        属性值 2
        ...
        属性值 n
        complete
        
        """

        # todo 自己增加了禁止输出的相关格式的案例



        return prompt

    # ==========================================
    # [修改] 核心扩展逻辑：集成 Sampler 和 DecisionMaker
    # ==========================================

    async def generate_samples_async(self):
        """
        [修改] 改为调用 IterativeSampler 进行迭代生成
        """
        base_prompt = self.format_gen_prompt()

        # 调用 Sampler
        # 初始时 samples 为空
        # Sampler 会处理 mini-batch, embedding 检查, 饱和度停止
        final_samples, is_saturated = await self.llm_engine.sampler.generate_until_saturated(
            base_prompt, existing_samples=[]
        )

        self.samples = final_samples
        self.logging(f"Generated {len(self.samples)} samples. Saturated: {is_saturated}", level="info")
        return self.samples, is_saturated

    def select_dimension_and_classify(self):
        """
                [同步逻辑] 调用 LLM 识别维度并分类样本。
                注意：实际主要使用下方的 async 版本，这个可能是保留的旧代码或用于调试。
        """
        parent_dimensions = self.retrieve_parent_dimensions()  # 获取完整路径划分准则

        prompt = self.format_dim_prompt()  # 获取提示词,其中逻辑已经获取完整路径划分准则和对应属性
        response = self.llm_engine.generate_batch(prompt, max_tokens=1024, temperature=1.0)  # 获取LLM的回答
        candidates = parse_json_candidates(response, logger=self.logger, debug=True)

        self.logging(f"Parsed candidates: {candidates}", level="debug")

        valid = True
        new_dim = None
        for c in candidates:
            if isinstance(c, dict) and ("dimension" in c) and ("attributes" in c):
                dim = c["dimension"].strip()
                attr_map = c["attributes"]

                all_indices = set()
                for cat, arr in attr_map.items():
                    if not isinstance(arr, list) or any(
                            not (isinstance(x, int) and 1 <= x <= len(self.samples))
                            for x in arr
                    ):
                        valid = False
                        self.logging(
                            f"Invalid attribute '{cat}': {arr}.", level="debug"
                        )
                        break
                    all_indices.update(arr)

                if valid and all_indices == set(range(1, len(self.samples) + 1)):
                    if dim not in parent_dimensions:
                        new_dim = c
                        break
                    else:
                        self.logging(
                            f"Dimension '{dim}' is already used.", level="debug"
                        )
        if new_dim is None:
            self.logging(
                "No valid dimension classification found. Generate again.",
                level="warning",
            )
            return self.select_dimension_and_classify()

        return new_dim

    async def select_dimension_and_classify_async(self, max_attempts=5, temp_excluded_dims=None):
        """
        todo 修改以兼容生成多个候选维度
        [异步分类] 让 LLM 决定如何分裂当前节点, 即 选择 划分准则 和 属性
        包含  重试机制，直到 LLM 返回合法的 JSON 且逻辑正确（所有样本都被分类，且没有遗漏）。

        :param temp_excluded_dims: (List[str]) 临时需要排除的维度（用于 MCTS 生成多个候选时去重）
        """

        if temp_excluded_dims is None:
            temp_excluded_dims = []

        for attempt in range(max_attempts):
            parent_dimensions = self.retrieve_parent_dimensions()  # 完整路径的 划分准则
            # todo [关键] 合并父路径排除项 + 临时排除项
            current_excluded = parent_dimensions + temp_excluded_dims

            prompt = self.format_dim_prompt(override_excluded=current_excluded)  # 生成 划分准则 和 属性 的提示词


            # 调用 LLM，temperature=1.0 鼓励多样性
            responses = await self.llm_engine.generate_batch_async(
                prompt, max_tokens=1024, temperature=1.0
            )
            response = responses[0] if isinstance(responses, list) else responses
            # 解析 LLM 返回的 JSON
            candidates = parse_json_candidates(response, logger=self.logger, debug=True)  # 返回列表
            """
            [
                { 
                "dimension": "维度名称", 
                "attributes": { 
                "属性1": [样本索引列表], 
                "属性2": [样本索引列表], ... 
                }
            ...包含一些[样本索引列表]的混合元素列表，后面会进行过滤
            ]
            """
            self.logging(
                f"[Attempt {attempt + 1}/{max_attempts}] Parsed dimension candidates: {candidates}",
                level="debug"
            )

            found_dim = None
            for c in candidates:
                valid = True
                dim = c["dimension"].strip()
                raw_attr_map = c["attributes"]  # 原始属性字典

                # [新增] 用于存储清洗后的属性字典
                cleaned_attr_map = {}
                all_indices = set()

                for cat, arr in raw_attr_map.items():
                    # ==========================================================
                    # [新增] 清洗 Key 的逻辑
                    # 正则解释：^ (开头) [\s\-\.\*\·\•\d]+ (匹配一个或多个空格、横杠、点、星号、中间点、圆点、数字)
                    # 这将去除 "- 属性", "· 属性", "1. 属性" 前面的修饰符
                    # ==========================================================
                    clean_cat = re.sub(r'^[\s\-\.\*\·\•\d]+', '', cat).strip()

                    # 校验 Value (保持原逻辑)
                    # 1) arr不是列表，或者 2) arr中有无效元素
                    if not isinstance(arr, list) or any(
                            not (isinstance(x, int) and 1 <= x <= len(self.samples))
                            for x in arr
                    ):
                        valid = False
                        self.logging(f"Invalid attribute '{cat}' (value check): {arr}.", level="debug")
                        break

                    # 如果 Key 被洗空了（例如原本Key就是个"-"), 则保留原Key防止丢失数据，否则使用新Key
                    final_key = clean_cat if clean_cat else cat

                    # 存入新字典
                    cleaned_attr_map[final_key] = arr
                    all_indices.update(arr)

                if valid:  # 校验逻辑完整性：所有样本是否都被覆盖？维度名是否重复？
                    if all_indices == set(range(1, len(self.samples) + 1)) \
                            and dim not in parent_dimensions \
                            and dim not in temp_excluded_dims:  # todo [新增] 检查临时排除

                        # [关键] 将清洗后的字典回写到 candidate 对象中
                        # 这样后续 MCTS 或 Expand 流程拿到的就是干净的 "属性1", "属性2"
                        c["attributes"] = cleaned_attr_map

                        found_dim = c
                        break

            if found_dim is not None:
                return found_dim  # 成功找到有效 划分准则和属性
            else:
                self.logging(
                    f"No valid dimension classification found in attempt {attempt + 1}, will retry...",
                    level="warning"
                )

        self.logging(
            f"Failed to classify dimension after {max_attempts} attempts => skip node expansion.",
            level="error"
        )
        return None

    async def expand_dimension_async(self, dimension, attribute_values):
        """
        [异步扩展 - 嵌入增强版]
        利用 Embedding 模型过滤掉与现有属性语义重复的生成项。
        todo 增加清理 排序格式的错误，返回的是清洗过且去重过的干净列表
        """
        # 0. [新增] 定义清洗函数
        def clean_attr_str(text):
            # 去除开头的 数字、点、横杠、星号、圆点 等符号
            return re.sub(r'^[\s\-\.\*\·\•\d]+', '', text).strip()
        # 1. 初始化配置
        config = self.llm_engine.config
        similarity_threshold = config.get("attr_similarity_threshold", 0.90)
        max_stuck = config.get("max_expand_attempts_stuck", 3)
        stuck_counter = 0 # 连续未产出有效新属性的次数

        # [新增] 清洗输入的初始属性列表
        # 这一步是为了防止传入的历史数据本身就不干净，影响 Embedding 计算
        attribute_values = [clean_attr_str(a) for a in attribute_values if clean_attr_str(a)]

        # 2. 准备 Sampler 和 初始 Embeddings
        sampler = self.llm_engine.sampler

        # 计算现有属性的 Embeddings (作为查重底库)
        # 注意：这里我们维护一个 embeddings 列表，随着新属性的加入动态更新
        current_embeddings = await sampler.get_embeddings_async(attribute_values)



        attempts = 0
        while True:
            prompt = self.format_expand_prompt(dimension, attribute_values)
            responses = await self.llm_engine.generate_batch_async(prompt, max_tokens=1024, temperature=0.7)
            raw_res = responses[0] if isinstance(responses, list) else responses
            # 解析每一行属性
            candidates = parse_attributes_from_str(raw_res, logger=self.logger, debug=True)
            # 校验基本格式
            if (not candidates) or (candidates[-1] not in ["null", "infinite", "complete"]):
                attempts += 1
                self.logging(f"Attempt {attempts}: invalid response => retrying", level="warning")
                continue# 格式错误，重试

            flag = candidates[-1]  # "null", "infinite", or "complete"
            if flag == "null":
                self.logging(f"LLM indicates dimension '{dimension}' is full (null).", level="info")
                break

            # 提取 LLM 建议的新属性 (排除最后一行的 flag)
            raw_proposed_attrs = candidates[:-1]

            if not raw_proposed_attrs:#防止LLM出现幻觉
                # 只有 flag 没有内容？
                if flag == "complete": break
                continue

            # [新增] 立即清洗新生成的属性
            proposed_attrs = []
            for attr in raw_proposed_attrs:
                cleaned = clean_attr_str(attr)
                if cleaned:  # 确保清洗后不是空字符串
                    proposed_attrs.append(cleaned)
            if not proposed_attrs:
                continue

            # ==========================================================
            # [核心逻辑] Embedding 查重
            # ==========================================================
            # 获取新候选的向量
            proposed_embeddings = await sampler.get_embeddings_async(proposed_attrs)

            # 利用 filter_new_samples 进行过滤
            # 参数: (新向量, 历史向量库, 新文本, 阈值)
            valid_attrs, valid_embs, has_duplicate = sampler.filter_new_samples(
                proposed_embeddings,
                current_embeddings,
                proposed_attrs,
                similarity_threshold
            )
            # 6. 处理过滤结果
            if valid_attrs:
                # 有新东西！
                attribute_values.extend(valid_attrs)
                current_embeddings.extend(valid_embs)
                self.logging(f"Expanded {len(valid_attrs)} new attributes: {valid_attrs}", level="info")

            # [核心修改] 饱和计数逻辑
            if has_duplicate:
                # 只要这批里有至少一个重复的，就记一次过错
                stuck_counter += 1
                self.logger.info(f"Duplicate detected in batch ({stuck_counter}/{max_stuck}).")
            else:
                # 如果这批全是新的，说明空间还很广阔，重置计数器
                stuck_counter = 0

            # 7. 安全熔断与循环控制

            # A. 数量超限熔断
            if len(attribute_values) > self.max_attribute_count:
                self.logging(f"Max attribute count ({self.max_attribute_count}) reached.", level="warning")
                break

            # B. 连续卡顿熔断 (LLM 一直在生成重复内容)
            if stuck_counter >= max_stuck:
                self.logging(f"Expansion stuck (repeated duplicates) for {max_stuck} rounds. Force stop.",
                             level="warning")
                break

            # C. LLM 自身标记结束
            if flag == "complete":
                self.logging("LLM marked expansion as complete.", level="info")
                break

            # D. flag == "infinite"，继续循环
            # (如果 valid_attrs 为空但 flag 是 infinite，我们会给它重试机会，直到 stuck_counter 触发)



            # todo 修改成迭代小批次生成 验证embedding相似度,注释掉的是原版
            # elif candidates[-1] == "infinite": #todo 这里已经用到挤牙膏的思想了，以实现更加多样性
            #     # LLM 认为属性还有很多，先加进去，然后继续循环生成
            #     # candidates[:-1] 的意思是：取列表里除了最后一个元素（"infinite"）之外的所有东西。
            #     # 这一步把 LLM 这次吐出来的 10 个新属性，追加到总列表 attribute_values 里。
            #     attribute_values += candidates[:-1]
            #     # 2. 安全熔断机制 (Safety Brake)
            #     # 这一步非常关键。如果没有这一步，如果 max_attribute_count 设得很大，
            #     # 或者 LLM 一直傻乎乎地回 infinite，程序就会陷入死循环，直到内存爆炸。
            #     if len(attribute_values) > self.max_attribute_count:
            #         break# 强制截断，防止无限循环
            #     else:
            #         self.logging("Insufficient attributes for infinite => continue refilling", level="warning")
            #         continue# 继续生成下一批
            # elif candidates[-1] == "null":
            #     # LLM 认为现有的属性已经够了
            #     self.logging(
            #         f"No valid expansion info found for dimension '{dimension}', use original attributes.",
            #         level="warning",
            #     )
            #     break
            # elif candidates[-1] == "complete":
            #     # LLM 补充了一些属性，并表示结束了
            #     attribute_values += candidates[:-1]
            #     break
            # else:
            #     self.logging(f"Unexpected last candidate: {candidates[-1]}", level="warning")
            #     continue

        return attribute_values#当前节点扩展后的属性列表，返回的是清洗过且去重过的干净列表
    async def expand_nodes_async(self, output_file=None, result_file=None):
        # 保存当前的树结构到文件，作为初始状态备份
        self.save_tree_structure(self.tree_structure_file)

        # 扩展根节点（或当前节点），获取第一批子节点
        # await 等待异步操作完成
        children = await self._expand_single_node_async(output_file, result_file)
        queue = deque(children)

        level = 0
        while queue:
            level_size = len(queue)
            tasks = []
            self.logging(f"[BFS] Start processing level={level} with {level_size} nodes", level="info")

            for _ in range(level_size):
                node = queue.popleft()
                tasks.append(asyncio.create_task(
                    node._expand_single_node_async(output_file, result_file)
                ))
            results = await asyncio.gather(*tasks)

            # BFS: collect next level
            for child_list in results:
                for c in child_list:
                    queue.append(c)
            self.save_tree_structure(self.tree_structure_file)
            level += 1

        self.save_tree_structure(self.tree_structure_file)

    async def _expand_single_node_async(self, output_file, result_file):
        """
        [融合版单节点逻辑]
        框架：遵循标准模板 (Max Depth -> Pivot -> Leaf Check -> Save)
        内核：集成 MCTS (候选生成 -> 预扩展 -> 模拟决策)
        """
        config = getattr(self.llm_engine, "config", {})
        infinite_path_samples = config.get("infinite_path_samples", 3)

        # =========================================================================
        # Phase 1: Max Depth Check (标准模板逻辑)
        # =========================================================================
        if self.depth >= self.max_depth:
            self.logging(f"[Leaf@MaxDepth] depth={self.depth}, stop expansion.", level="info")

            infinite_count = self.count_infinite_nodes_in_path()

            # [正确] 计算目标轮次
            target_batches = max(1, infinite_path_samples ** infinite_count)

            # [正确] 判定已完成的轮数
            # 如果 self.samples 非空，说明 MCTS 模拟阶段已经跑过 1 轮
            batches_completed = 1 if (self.samples and len(self.samples) > 0) else 0

            # [正确] 计算还需要跑几轮
            batches_needed = target_batches - batches_completed

            if batches_needed > 0:
                self.logging(
                    f"Path has {infinite_count} infinite nodes. Target rounds: {target_batches}. "
                    f"Executed: {batches_completed}. Running {batches_needed} more rounds.",
                    level="info")

                # [关键修正]
                # 不需要手动维护 all_samples = []，也不需要 extend。
                # 因为 generate_samples_async 内部会自动读取 self.samples，生成新数据，并追加更新 self.samples。
                # 我们只需要“空转”触发它即可。
                for i in range(batches_needed):
                    self.logging(f"Executing compensation batch {i + 1}/{batches_needed}...", level="info")
                    await self.generate_samples_async()

            # [兜底] 确保 self.samples 至少是个列表（防止 NoneType Error）
            if self.samples is None:
                self.samples = []

            # 保存结果 (标准模板逻辑)
            if result_file and self.samples:
                with open(result_file, "a", encoding="utf-8") as f:
                    for q in self.samples:
                        line = {"question": q}
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")

            # 保存树结构 (标准模板逻辑)
            if output_file:
                tree_dict = self.retrieve_root().to_dict()
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(tree_dict, f, ensure_ascii=False, indent=2)

            return []

        # =========================================================================
        # Phase 2: Pivot Sample Generation (MCTS 增强版)
        # =========================================================================
        # MCTS 逻辑注入：检查是否可以复用模拟阶段的样本
        if self.samples and len(self.samples) > 0:
            self.logging(f"Node already has {len(self.samples)} samples (from MCTS). Skipping initial generation.",
                         level="info")
            samples = self.samples
        else:
            # 标准逻辑：生成枢轴样本
            samples, is_saturated = await self.generate_samples_async()

        # MCTS 逻辑注入：空间狭窄检查 (Saturated Check)
        min_samples = config.get("min_samples_per_node", 5)
        if len(samples) < min_samples:
            self.logging(f"[Leaf@Saturated] Only {len(samples)} samples. Stop splitting.", level="info")

            self.children = []
            # 跳转到 Phase 4 直接进入叶子节点处理的逻辑

        else:
            candidates_json = []  # 存储所有收集到的候选维度对象
            # 优先判断能不能生成 划分准则和分类属性,用于叶子节点的判断，兼容未融入MCTS的处理逻辑, todo 如果能够生成，后面生成候选 基于first_dim_dict
            first_dim_dict = await self.select_dimension_and_classify_async(max_attempts=5)
            # 如果尝试多次 LLM 还是无法分类（比如样本太杂乱），则放弃分裂，将当前节点降级为叶子节点。
            if first_dim_dict is None:
                self.logging("Dimension classification failed => treat this node as leaf.", "warning")

                self.children = []
            else:
                # =========================================================================
                # Phase 3: Dimension Selection & Expansion (MCTS 替换核心)
                # 原逻辑：select_dimension -> expand
                # 新逻辑：generate candidates -> pre-expand all -> decide -> pick winner
                # =========================================================================

                # # 3.1 Candidate Generation
                # n_candidates = config.get("n_candidate_dimensions", 2)
                # mcts_prompt = self.llm_engine.format_mcts_dim_prompt(samples, self.retrieve_parent_dimensions(),
                #                                                      n_candidates)
                # responses = await self.llm_engine.generate_batch_async(mcts_prompt, temperature=1.0)
                # candidates_json = parse_json_candidates(responses[0], self.logger)
                #



                # todo 复用原方法 迭代生成候选
                candidates_json.append(first_dim_dict)
                first_dim_name = first_dim_dict["dimension"]
                self.logging(f"First candidate found: {first_dim_name}", level="info")

                # 后续轮次生成 (MCTS 需要 N 个候选)
                n_candidates = config.get("n_candidate_dimensions", 2)
                if n_candidates > 1:
                    # 已经有 1 个了，还需要生成 n-1 个
                    # 我们采用串行生成，因为每一轮都要排除上一轮的结果
                    current_excluded = [first_dim_name]

                    for i in range(n_candidates - 1):
                        self.logging(f"Generating candidate {i + 2}/{n_candidates}...", level="info")

                        # 调用分类函数，传入当前已有的维度作为排除项
                        next_dim = await self.select_dimension_and_classify_async(
                            max_attempts=5,  # 可以适当减少尝试次数，比如 3
                            temp_excluded_dims=current_excluded
                        )

                        if next_dim:
                            candidates_json.append(next_dim)
                            current_excluded.append(next_dim["dimension"])
                            self.logging(f"Candidate {i + 2} found: {next_dim['dimension']}", level="info")
                        else:
                            self.logging(f"Failed to generate candidate {i + 2}. Stop finding more.", level="warning")
                            break  # 生成不出来就算了，有多少用多少



                if not candidates_json:
                    self.logging("No valid candidates found.", level="warning")

                    self.children = []  # 标记为无子节点，触发 Leaf 逻辑 ，跳到
                else:
                    # 3.2 Pre-process Candidates (构造节点组)
                    candidate_node_groups = []#候选节点组

                    # 临时保存 children 以防被覆盖，虽然此时应该是空的
                    temp_all_potential_children = []

                    for cand in candidates_json:
                        dim_name = cand.get("dimension")
                        initial_attrs = list(cand.get("attributes", {}).keys())

                        # Pre-Expansion (立即扩展属性)
                        self.logging(f"Expanding attributes for candidate '{dim_name}'...", level="info")
                        full_attrs = await self.expand_dimension_async(dim_name, initial_attrs)

                        current_group = []

                        # Infinite Check (立即判定拓扑结构)
                        if self.is_infinite(full_attrs):
                            # 创建聚合节点
                            child = TreeNode(
                                depth=self.depth + 1,
                                llm_engine=self.llm_engine,
                                dimension=dim_name,
                                attribute_value=full_attrs,
                                parent=self,
                                is_selected=False,
                                diversity_score=0.0,
                                max_depth=self.max_depth,
                                num_samples_per_node=self.num_samples_per_node,
                                infinite_threshold=self.infinite_threshold,
                                max_attribute_count=self.max_attribute_count,
                                threadpool_executor=self.threadpool_executor,
                                tree_structure_file=self.tree_structure_file  # 保持传递文件名
                            )
                            current_group.append(child)
                            temp_all_potential_children.append(child)
                        else:
                            # 创建普通子节点
                            for attr in full_attrs:
                                child = TreeNode(
                                    depth=self.depth + 1,
                                    llm_engine=self.llm_engine,
                                    dimension=dim_name,
                                    attribute_value=attr,
                                    parent=self,
                                    is_selected=False,
                                    diversity_score=0.0,
                                    max_depth=self.max_depth,
                                    num_samples_per_node=self.num_samples_per_node,
                                    infinite_threshold=self.infinite_threshold,
                                    max_attribute_count=self.max_attribute_count,
                                    threadpool_executor=self.threadpool_executor,
                                    tree_structure_file=self.tree_structure_file
                                )
                                current_group.append(child)
                                temp_all_potential_children.append(child)

                        candidate_node_groups.append(current_group)
                        """
                        [
                              [Child_A1, Child_A2, Child_A3],  # 第 0 组：维度 A (例如 Difficulty) 的所有孩子
                              [Child_B1, Child_B2]             # 第 1 组：维度 B (例如 Topic) 的所有孩子
                        ]
                        """

                    # 3.3 MCTS Decision
                    # 将构建好的节点组交给 MCTS 进行模拟和打分
                    # 注意：这里需要暂时把 self.children)，以便 output.json 能记录所有尝试
                    self.children = temp_all_potential_children

                    active_children = await self.llm_engine.decision_maker.decide_best_split(self, candidate_node_groups)

                    # 3.4 Finalize Children
                    # 将 self.children 更新为所有候选节点（包含被淘汰的，状态为 is_selected=False）
                    # 但函数返回值只返回 active_children
                    if not active_children:
                        self.logging("MCTS returned no active children.", level="warning")

                        # self.children = []  # 里面全是 is_selected=False 的节点），Phase 4 依然会正确触发叶子兜底逻辑。
                        # Phase 4 和 Phase 5 保存 output.json / tree_structure.txt 时，你就能看到 MCTS 尝试过但全军覆没的那些维度和得分。
                    else:
                        # 关键：这里不需要把 self.children 设为 active_children
                        # self.children 应该保留所有节点(为了 tree_structure 记录)，
                        # 而 BFS 只需要处理 active 的。
                        # 后续方便保存 舍弃的路径
                        pass

                    # =========================================================================
        # Phase 4: Leaf Node Handling (标准模板逻辑 - 兜底)
        # 只要没有有效子节点（MCTS 失败、饱和、无候选），就视为叶子进行补偿生成
        # =========================================================================

        # 这里的判断逻辑微调：检查是否有 is_selected=True 的孩子
        has_active_children = any(c.is_selected for c in self.children)

        if not has_active_children:
            self.logging(f"[Leaf] Node dimension={self.dimension} => leaf node (or failed split), writing samples.",
                         level="info")

            # 标记为叶子节点 直接执行最终的生成
            # todo 自己修改 执行无限节点步长
            infinite_count = self.count_infinite_nodes_in_path()

            # [正确] 计算目标轮次
            target_batches = max(1, infinite_path_samples ** infinite_count)

            # [正确] 判定已完成的轮数
            # 如果 self.samples 非空，说明 MCTS 模拟阶段已经跑过 1 轮
            batches_completed = 1 if (self.samples and len(self.samples) > 0) else 0

            # [正确] 计算还需要跑几轮
            batches_needed = target_batches - batches_completed

            if batches_needed > 0:
                self.logging(
                    f"Path has {infinite_count} infinite nodes. Target rounds: {target_batches}. "
                    f"Executed: {batches_completed}. Running {batches_needed} more rounds.",
                    level="info")

                # [关键修正]
                # 不需要手动维护 all_samples = []，也不需要 extend。
                # 因为 generate_samples_async 内部会自动读取 self.samples，生成新数据，并追加更新 self.samples。
                # 我们只需要“空转”触发它即可。
                for i in range(batches_needed):
                    self.logging(f"Executing compensation batch {i + 1}/{batches_needed}...", level="info")
                    await self.generate_samples_async()

            # [兜底] 确保 self.samples 至少是个列表（防止 NoneType Error）
            if self.samples is None:
                self.samples = []

            # 保存结果 (标准模板逻辑)
            if result_file and self.samples:
                with open(result_file, "a", encoding="utf-8") as f:
                    for q in self.samples:
                        line = {"question": q}
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")

            # 保存树结构 (标准模板逻辑)
            if output_file:
                tree_dict = self.retrieve_root().to_dict()
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(tree_dict, f, ensure_ascii=False, indent=2)

            # 返回空列表，停止 BFS
            return []

        # =========================================================================
        # Phase 5: Final Save & Return (标准模板逻辑)
        # =========================================================================
        if output_file:
            tree_dict = self.retrieve_root().to_dict()
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(tree_dict, f, ensure_ascii=False, indent=2)

        # 只返回被选中的孩子给 BFS 队列
        active_children = [c for c in self.children if c.is_selected]
        return active_children

    @staticmethod
    def from_dict(d: dict, parent=None):
        node = TreeNode(
            depth=0,
            llm_engine=None,
            threadpool_executor=None,
            dimension=d.get("dimension"),
            attribute_value=d.get("attribute_value"),
            max_depth=5,
            num_samples_per_node=10,
            infinite_threshold=50,
            max_attribute_count=50,
            parent=parent,
            tree_structure_file="tree_structure.txt",
        )
        node.samples = d.get("samples", [])

        for child_dict in d.get("children", []):
            child_node = TreeNode.from_dict(child_dict, parent=node)
            node.children.append(child_node)

        if parent:
            node.depth = parent.depth + 1

        return node
    def _save_results(self, result_file):
        """辅助函数：保存样本到 result file"""
        if result_file and self.samples:
            with open(result_file, "a", encoding="utf-8") as f:
                for q in self.samples:
                    line = {"question": q}
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")

    # ... (保持 expand_nodes_async, from_dict, inject_runtime_to_tree 等函数不变) ...


# ==========================================
# 辅助函数 (保持不变)
# ==========================================
def parse_json_candidates(response, logger=None, debug=False):

    if not response or not isinstance(response, str):
        if logger:
            logger.error("Invalid response for JSON parsing.")
        return []

    if response.strip().lower() == '"infinite"':
        if logger:
            logger.info("Detected 'infinite' response.")
        return "infinite"
    # return "infinite"

    results = []

    def extract_balanced(text, open_char, close_char):
        stack = []
        start = None
        candidates = []
        for i, char in enumerate(text):
            if char == open_char:
                if not stack:
                    start = i
                stack.append(char)
            elif char == close_char and stack:
                stack.pop()
                if not stack:
                    candidates.append(text[start : i + 1])
                    start = None
        return candidates

    obj_candidates = extract_balanced(response, "{", "}")
    arr_candidates = extract_balanced(response, "[", "]")
    candidates = obj_candidates + arr_candidates

    for json_str in candidates:
        cleaned_str = json_str.strip().rstrip(",").rstrip(".").strip()
        try:
            parsed = json.loads(cleaned_str)
            if isinstance(parsed, (dict, list)):
                results.append(parsed)
        except json.JSONDecodeError:
            lines = response.splitlines()
            results = [line.strip() for line in lines if line.strip()]
            if debug and logger:
                logger.debug(f"JSONDecodeError: {cleaned_str}")

    if not results and logger:
        logger.warning("No valid JSON found.")
    else:
        if logger and debug:
            logger.debug(f"Parsed JSON candidates: {results}")

    return results


def parse_attributes_from_str(response: str, logger=None, debug=False) -> list:

    if logger and debug:
        logger.debug(f"Step3 raw response:\n{response}")

    lines = response.splitlines()
    lines = [l.strip() for l in lines if l.strip()]

    if logger and debug:
        logger.debug(f"Parsed lines: {lines}")

    if len(lines) == 1 and lines[0].lower() == "null":
        if logger and debug:
            logger.debug(
                "LLM indicates existing attributes fully cover the dimension (null)."
            )
        return ["null"]

    attr_values = [x.strip('"') for x in lines]
    return attr_values

def inject_runtime_to_tree(node: TreeNode,
                           llm_engine,
                           threadpool_executor,
                           max_depth,
                           infinite_threshold,
                           max_attribute_count):

    node.llm_engine = llm_engine
    node.threadpool_executor = threadpool_executor
    node.max_depth = max_depth
    node.infinite_threshold = infinite_threshold
    node.max_attribute_count = max_attribute_count

    for child in node.children:
        child.parent = node
        child.depth = node.depth + 1
        inject_runtime_to_tree(child,
                               llm_engine,
                               threadpool_executor,
                               max_depth,
                               infinite_threshold,
                               max_attribute_count)


def setup_logger(log_file="generation.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
