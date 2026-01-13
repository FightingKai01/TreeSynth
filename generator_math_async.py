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
import asyncio,threading
import httpx
import random
import concurrent.futures
import time
from config import *

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
    """Unified inference class for both vLLM and OpenAI backends"""
    
    def __init__(
        self,
        backend="vllm",
        api_pool=None,
        config=None,
        logger=None,
        max_retries=5,
        max_workers=20,
        max_concurrent_requests=64,
    ):
        self.logger = logger or logging.getLogger()#短路取值
        self.backend = backend.lower().strip()
        self.max_retries = max_retries
        self.threadpool_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        )#创建线程池
        self.api_pool = api_pool
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)#asyncio.Semaphore（异步信号量）就是专门用来控制并发数量的工具
        self.config = config or {}#配置参数是字典类型

        if self.backend == "vllm":
            self.model_name = self.config.get("model_name")
            if api_pool is None:#无API pool
                self.logger.info("Using vLLM backend with single endpoint.")
                self.client = OpenAI(
                    base_url=self.config.get("api_base"), 
                    api_key=self.config.get("api_key")
                )
            else:
                self.logger.info("Using vLLM backend with API pool.")
                # Client will be created per request from the pool

        elif self.backend == "azure":
            self.model_name = self.config.get("model_name", "gpt-4o")
            if api_pool is None:
                self.logger.info("Using Azure backend with single endpoint.")
                self.client = AzureOpenAI(
                    azure_endpoint=self.config.get("azure_endpoint"),
                    api_key=self.config.get("api_key"),
                    api_version=self.config.get("api_version")
                )
            else:
                self.logger.info("Using Azure backend with API pool.")
                # Client will be created per request from the pool

        # todo 自行修改，源代码有问题.
        elif self.backend == "openai":
            self.model_name = self.config.get("model_name", "gpt-4o")
            if api_pool is None:
                self.logger.info("Using OpenAI backend with single endpoint.")
                self.client = OpenAI(
                    base_url=self.config.get("api_base"),
                    api_key=self.config.get("api_key")
                )
            else:
                self.logger.info("Using OpenAI backend with API pool.")
                # Client will be created per request from the pool
        else:
            raise ValueError(f"backend must be 'vllm' or 'openai', got {self.backend}.")
    
    async def generate_per_prompt_async(self, prompt, max_tokens=1024, temperature=0.7):
        """
                异步执行单个 Prompt 的生成任务。
                包含并发限制、API 轮询池支持、自动重试和错误处理机制。

                Args:
                    prompt (str): 输入给大模型的提示词文本。
                    max_tokens (int, optional): 模型生成的最大 Token 数限制。默认为 1024。
                    temperature (float, optional): 采样温度，控制生成的随机性 (0.0~1.0)。默认为 0.7。

                Returns:
                    str: 模型生成的文本内容。如果多次重试后仍失败，则返回空字符串 ""。
        """
        # [参数校验] 确保 prompt 是字符串类型，防止传入非字符串导致后续拼接或 API 调用报错
        assert isinstance(prompt, str), "Prompt must be a string."
        # [初始化状态变量]
        retries = 0  # 当前已重试的次数
        ans = ""  # 用于存储最终结果的变量，初始化为空字符串
        # [初始化退避时间]
        # 初始等待时间为 1.0 秒。采用"指数退避" (Exponential Backoff) 策略：
        # 失败一次等 1s，再失败等 2s，再失败等 4s... 以避免在服务器拥堵时加剧负载。
        backoff = 1.0
        while retries < self.max_retries:
            retries += 1
            # [并发控制 - 关键点]
            # self.semaphore 是一个 asyncio.Semaphore 对象 (信号量)。
            # async with 会在这里排队等待，直到获得一个"令牌"。
            # 只有拿到令牌的代码才能进入下方的代码块执行，从而限制同时发起的 API 请求数量，
            # 防止因并发过高导致显存溢出 (OOM) 或触发 API 服务商的 Rate Limit。
            async with self.semaphore:
                try:
                    # [API Pool 负载均衡逻辑]
                    # 如果启用了 API Pool (self.api_pool 不为 None)，说明我们要从多个 Key/Endpoint 中轮询挑选一个。
                    if self.api_pool:
                        # Select from pool based on backend
                        if self.backend == "vllm":
                            # [vLLM 后端]
                            # get_next_config() 会以轮询 (Round-Robin) 方式返回一个可用的配置字典和其索引
                            conf, conf_idx = self.api_pool.get_next_config()
                            # 动态创建一个临时的 OpenAI 客户端实例
                            # 因为每次轮询到的 base_url 和 api_key 可能不同，所以不能复用全局 client
                            local_client = OpenAI(
                                base_url=conf["api_base"],
                                api_key=conf["api_key"]
                            )
                            model_name = conf["model_name"]
                        elif self.backend == "azure": # [Azure 后端]
                            conf, conf_idx = self.api_pool.get_next_config()
                            local_client = AzureOpenAI(
                                azure_endpoint=conf["endpoint"],
                                api_key=conf["key"],
                                api_version=conf["version"]
                            )
                            model_name = conf["model"]
                        elif self.backend == "openai": # todo 自己修改 增加 openai的客户端配置 同 vLLM配置一样
                            # [openai 后端]
                            # get_next_config() 会以轮询 (Round-Robin) 方式返回一个可用的配置字典和其索引
                            conf, conf_idx = self.api_pool.get_next_config()
                            # 动态创建一个临时的 OpenAI 客户端实例
                            # 因为每次轮询到的 base_url 和 api_key 可能不同，所以不能复用全局 client
                            local_client = OpenAI(
                                base_url=conf["api_base"],
                                api_key=conf["api_key"]
                            )
                            model_name = conf["model_name"]
                    else:
                        # [单一客户端模式]
                        # 如果没启用池子，直接 复用 类初始化时创建好的全局 client 和模型名
                        local_client = self.client
                        model_name = self.model_name
                    # [输入日志记录]
                    # 记录即将发送的请求详情，方便调试和追踪
                    if self.logger:
                        self.logger.info(
                            f"[Async LLM Input] model={model_name}, tokens={max_tokens}, temp={temperature}, Prompt:\n{prompt}"
                        )
                    # [执行 API 调用 - 核心难点]
                    # local_client.chat.completions.create 是一个同步 (阻塞) 函数！
                    # 如果在 async 函数里直接调用它，会阻塞整个 asyncio 事件循环，导致其他并发任务全部暂停。
                    #
                    # asyncio.to_thread 的作用：
                    # 将这个同步的 API 请求扔到一个独立的线程池 (ThreadPool) 中去运行，
                    # 这样主线程 (Event Loop) 可以立即释放去处理其他任务，从而实现真正的异步高并发。
                    # await 等待线程池返回结果。
                    completion = await asyncio.to_thread(
                        local_client.chat.completions.create,
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    """OpenAI API响应的结构
                    response
                    ├── choices (列表)
                    │   └── [0] (第一个选择，通常只有一个)
                    │       ├── message (消息对象)
                    │       │   ├── role (角色，如 "assistant")
                    │       │   └── content (内容，AI生成的文本)
                    │       └── finish_reason (结束原因)
                    └── usage (token使用统计)
                    """
                    # [提取结果]
                    # 解析返回的对象，获取第一个候选项的内容，并去除首尾空白字符
                    ans = completion.choices[0].message.content.strip()
                    # [输出日志记录]
                    if self.logger:
                        self.logger.info(f"[Async LLM API Output]:\n{ans}")
                    # [上报成功状态]
                    # 如果使用了 API Pool，必须通知池子这个节点 (conf_idx) 这次请求成功了。
                    # 池子可能会根据成功率来调整该节点的权重或将其标记为健康。
                    # Report success if using pool
                    if self.api_pool:
                        self.api_pool.report_success(conf_idx)
                        
                    break  # success => break# [成功退出] 拿到结果后，直接跳出 while 循环，不再重试
                # [异常处理 1: HTTP 状态错误]
                # httpx 是 OpenAI 库底层使用的 HTTP 客户端，用于捕获 4xx/5xx 错误
                except httpx.HTTPStatusError as http_err:
                    # Report error if using pool# 如果用了池子，先上报错误。池子可能会暂时禁用这个节点。
                    if self.api_pool:
                        self.api_pool.report_error(conf_idx)
                    # 特殊处理 429 (Too Many Requests - 请求过多/限流)
                    if http_err.response.status_code == 429:
                        self.logger.error(
                            f"[429] Too Many Requests => attempt {retries}/{self.max_retries}, backoff={backoff}s"
                        )
                        # [异步休眠]
                        # await asyncio.sleep 会挂起当前任务，把 CPU 让给其他任务，
                        # 等待 backoff 秒后再回来。
                        await asyncio.sleep(backoff)
                        backoff *= 2
                    else:
                        # 处理其他 HTTP 错误 (如 500 服务器内部错误, 401 认证失败等)
                        self.logger.error(f"HTTPStatusError: {http_err}, attempt {retries}.")
                        await asyncio.sleep(backoff)
                        backoff *= 2
                # [异常处理 2: 通用错误]
                # 捕获其他所有未预料的异常 (如网络断开、JSON 解析失败、超时等)
                except Exception as e:
                    # Report error if using pool
                    if self.api_pool:
                        self.api_pool.report_error(conf_idx)
                        
                    self.logger.error(f"Async LLM API error: {e}, attempt {retries}.")
                    await asyncio.sleep(backoff)
                    backoff *= 2
        # [最终失败检查]
        # 如果循环结束了 (超过最大重试次数)，ans 依然是空字符串，说明任务彻底失败
        if ans == "":
            self.logger.error(f"[Failed] No response after {self.max_retries} attempts.")
        # [返回结果]
        # 返回提取到的文本，如果失败则返回 ""
        return ans

    async def generate_batch_async(self, prompts, max_tokens=1024, temperature=0.7):
        """
        Asynchronously generate completions for multiple prompts.
        """
        """
                [批量生成入口] 异步地为多个 Prompt 生成回复。

                这个函数是实现"高吞吐量"的关键。它不再是一个个地等结果，
                而是同时发出所有请求，然后统一回收结果。

                Args:
                    prompts (str or List[str]): 可以是一个单独的字符串 Prompt，也可以是 Prompt 列表。
                    max_tokens (int): 生成的最大长度。
                    temperature (float): 采样温度。

                Returns:
                    str or List[str]: 
                        - 如果输入是单个字符串，返回单个字符串结果。
                        - 如果输入是列表，返回结果列表 (顺序与输入严格一致)。
        """
        # 1. [多态处理] 兼容单个字符串输入
        # 有时候调用者可能只想发一条，为了接口友好性，这里做了类型判断。
        if isinstance(prompts, str):
            # If only one prompt, directly call single prompt\
            # 直接 await 调用单条处理函数，不需要创建 Task 列表的开销
            return await self.generate_per_prompt_async(prompts, max_tokens, temperature)#返回字符串

        # 2. [任务列表初始化]
        # 用于存放所有的 Task 对象。Task 对象代表了"正在后台运行"的任务。
        tasks = []
        # 3. [任务分发循环]
        # 遍历输入的每一个 prompt
        for p in prompts:
            # 4. [创建并调度任务 - 关键步骤]
            # asyncio.create_task(...) 做两件事：
            #   a. 将协程 wrap 成一个 Task 对象。
            #   b. **立即**将该任务扔进事件循环 (Event Loop) 开始排队执行。
            # 注意：代码运行到这里不会阻塞（不会等任务做完），而是瞬间执行完循环，
            # 就像领导把 10 个文件分发给 10 个员工，发完就走，不站在员工旁边等
            tasks.append(
                asyncio.create_task(
                    self.generate_per_prompt_async(p, max_tokens, temperature)
                )
            )
        # Execute concurrently
        # 5. [并发等待与结果收集]
        # asyncio.gather(*tasks) 的作用：
        #   a. 挂起当前函数，直到列表里所有的 task 全部完成 (或者报错)。
        #   b. **保序**：它返回的 results 列表顺序，严格对应传入 tasks 的顺序。
        #      即使第 10 个任务比第 1 个任务先做完，结果里它依然排在第 10 位。
        # *tasks 是解包操作，把列表展开成参数传进去。
        results = await asyncio.gather(*tasks)
        # 6. [返回结果]
        # 返回的是一个字符串列表，对应输入的 prompts 列表
        return results

def parse_json_candidates(response, logger=None, debug=False):
    """
    鲁棒性极强的 JSON 解析器。
    它不信任 LLM 的输出是纯净的 JSON，而是假设输出中混杂了自然语言、Markdown 符号等杂质。
    """
    # 1. 基础防御：防止传入 None 或非字符串
    if not response or not isinstance(response, str):
        if logger:
            logger.error("Invalid response for JSON parsing.")
        return []
    # 2. 特殊指令拦截
    # 有时 Prompt 允许 LLM 输出 "infinite" 表示某种特殊状态，这里做拦截
    if response.strip().lower() == '"infinite"':
        if logger:
            logger.info("Detected 'infinite' response.")
        return "infinite"
    # 3. 存储解析结果的列表
    results = []

    # 4. 核心逻辑：提取平衡括号（解决嵌套问题）
    # 为什么不能用正则表达式？因为正则很难处理多层嵌套的 JSON，比如 {"a": {"b": 1}}
    # 使用栈（Stack）可以完美匹配最外层的左括号和对应的右括号。
    def extract_balanced(text, open_char, close_char):
        """
                从文本中提取平衡括号的内容。

                使用栈数据结构来匹配对应的括号，确保提取出完整的括号对。

                参数:
                    text (str): 要搜索的文本
                    open_char (str): 开括号字符，如 '{', '['
                    close_char (str): 闭括号字符，如 '}', ']'

                返回:
                    list: 包含所有完整括号对的字符串列表
        """
        stack = []  # 栈：用于记录目前还有几个左括号没被闭合
        start = None  # 指针：记录当前最外层左括号的索引
        candidates = []  # 结果箱：存找到的完整 JSON 字符串块

        # 遍历文本中的每个字符
        for i, char in enumerate(text):
            # 遇到开括号
            if char == open_char:
                if not stack:  # 如果栈为空，说明这是一个新的括号对开始
                    start = i  # 栈空时遇到的左括号，就是最外层的开始
                stack.append(char)  # 入栈

            # 遇到闭括号且栈不为空（有匹配的开括号）
            elif char == close_char and stack:
                stack.pop()  # 遇到右括号，消掉一个左括号（出栈）

                # 如果栈为空，说明找到了一个完整的括号对
                if not stack:
                    # 栈再次为空，说明最外层的括号闭合了！
                    # 提取这段完整的字符串（包含左右括号）
                    candidates.append(text[start: i + 1])
                    start = None  # 重置起始位置
        return candidates

    # 5. 双管齐下：既找字典 {} 也找列表 []
    obj_candidates = extract_balanced(response, "{", "}")
    arr_candidates = extract_balanced(response, "[", "]")
    # 合并两种类型的候选
    candidates = obj_candidates + arr_candidates#合并之后是一个列表（可包含混合元素）
    """解释为何要提取 []内容
    candidates = [
    # 第 1 个元素（来自 obj）：这是我们真正想要的完整大鱼
    '{ "dimension": "Difficulty", "attributes": { "Easy": [1, 2], ... } }',
    
    # 第 2 个元素（来自 arr）：这是无用的碎片
    '[1, 2]',
    
    # 第 3 个元素（来自 arr）：也是无用的碎片
    '[3, 4]'
    ]
    这得益于后续的清洗和校验逻辑select_dimension_and_classify_async:
    # 校验逻辑（这才是过滤器！）
    if isinstance(c, dict) and ("dimension" in c) and ("attributes" in c):
        # ... 处理逻辑 ...
        
    为什么要这么做？ 是为了防止 LLM 偶尔抽风，不按你的要求输出字典，而是直接输出一个列表（比如 [{"dimension":...}]）。虽然在你的这个特定 Prompt 格式下
    """
    # 6. 逐个尝试解析
    for json_str in candidates:
        # 清洗小尾巴：有些 LLM 会在 JSON 结尾手滑加个逗号或句号，会导致报错
        cleaned_str = json_str.strip().rstrip(",").rstrip(".").strip()# 清理字符串：去除首尾空白，移除末尾的逗号和句点
        try:
            # 尝试解析为JSON
            parsed = json.loads(cleaned_str)

            # 只有解析出来确实是 字典 或 列表 才是我们想要的
            if isinstance(parsed, (dict, list)):
                results.append(parsed)

        except json.JSONDecodeError:#json解析失败的原因，用于调试
            # 7. 降级策略（Fallback）
            # 如果提取出来的块都不是合法 JSON（极为罕见，可能是格式极度混乱），
            # 这里的逻辑是：放弃 JSON 解析，直接把原始文本按行切分返回。
            lines = response.splitlines()#字符串方法，按行分割文本,会自动处理不同的换行符：\n（Unix/Linux）、\r\n（Windows）、\r（旧Mac）
            results = [line.strip() for line in lines if line.strip()]#if line.strip() - 只保留非空行

            # 调试模式下记录解析失败的信息
            if debug and logger:
                logger.debug(f"JSONDecodeError: {cleaned_str}")
            break  # 一旦发生解析错误，停止尝试解析其他候选

    if not results and logger:
        logger.warning("No valid JSON found.")
    else:
        if logger and debug:
            logger.debug(f"Parsed JSON candidates: {results}")

    return results


def parse_expand_attributes_from_str(response: str, logger=None, debug=False) -> list:
    """
        它的主要任务是处理 “属性扩展” (Attribute Expansion) 阶段的 LLM 返回值。在这个阶段，Prompt 明确要求 LLM “按行输出” 新的属性值，并在最后一行加上 complete 或 infinite 或直接返回 null

        解析 LLM 返回的纯文本列表。
        预期格式是每行一个属性值，最后一行可能是状态标记（complete/infinite）。
    """
    # 1. 调试日志：记录原始输入
    if logger and debug:
        logger.debug(f"Step3 raw response:\n{response}")
    # 2. 核心逻辑：按行分割
    lines = response.splitlines()# response.splitlines() 会自动处理 \n, \r\n 等换行符
    # 3. 数据清洗
    # l.strip(): 去掉每行前后的空格
    # if l.strip(): 过滤掉空行（防止 LLM 输出很多空行导致解析出空字符串）
    lines = [l.strip() for l in lines if l.strip()]

    if logger and debug:
        logger.debug(f"Parsed lines: {lines}")
    # 4. 特殊指令处理：NULL
    # Prompt 中规定：如果现有属性已经涵盖了所有情况，LLM 应只输出 "null"
    # 这里检测：如果只有一行且内容是 null（忽略大小写），则直接返回 ["null"]
    # 上层逻辑收到 ["null"] 后会停止扩展。
    if len(lines) == 1 and lines[0].lower() == "null":
        if logger and debug:
            logger.debug(
                "LLM indicates existing attributes fully cover the dimension (null)."
            )
        return ["null"]
    # 5. 去除多余引号
    # 有些 LLM 比较“听话”但也比较啰嗦，它可能会输出："Attribute 1" (带双引号)。
    # 这里 x.strip('"') 专门用来把两头的双引号剥掉，还原纯文本。
    attr_values = [x.strip('"') for x in lines]
    # 注意：这里返回的列表可能包含 "complete" 或 "infinite" 作为最后一项。
    # 真正的逻辑判断（是继续生成还是结束）是在调用这个函数的 expand_dimension_async 里处理的。
    return attr_values#列表

class TreeNode:

    def __init__(
        self,
        depth,
        llm_engine=None,
        parent=None,
        dimension=None,
        attribute_value=None,
        max_depth=5,
        num_samples_per_node=10,
        infinite_threshold=10,
        max_attribute_count=20,
        threadpool_executor=None,
        tree_structure_file="tree_structure.txt",
    ):
        """
            树节点类：这是数据生成树的基本单元。
            每个节点代表数据空间的一个切片（例如：根 -> 代数 -> 线性方程 -> ...）。
        """
        # --- 基础完整性校验 ---
        if parent is None:
            # 如果没有父节点，说明是根节点，深度必须为 0
            assert depth == 0, "Root node must have depth=0 if no parent."
        else:
            # 这里的深度必须是父节点深度 + 1
            assert depth == parent.depth + 1, "Child node must have parent's depth+1."
        self.depth = depth # 当前节点在树中的层级

        # --- 核心属性 ---
        self.dimension = dimension  # 当前层划分的维度名称 (例如 "Topic", "Difficulty")
        self.attribute_value = attribute_value  # 当前节点对应的具体属性值 (例如 "Algebra", "Hard") 注意：每个节点都有自己唯一的属性值（其父节点 划分准则下的一个属性值， 该节点是父节点的孩子，并非属性值是孩子），如果当前节点的属性值是list或set，证明其父节点的划分准则下 属性值是无限的，即该节点是有个无限节点
        self.parent = parent  # 指向父节点的引用
        self.children = []  # 存储子节点的列表
        self.samples = None  # 存储当前节点生成的样本数据 (题目列表)
        self.dimensions = []

        # --- 配置参数 ---
        self.max_depth = max_depth  # 树生长的最大深度，超过此深度停止分裂
        self.num_samples_per_node = num_samples_per_node  # 每次生成多少个样本用于分析
        self.max_attribute_count = max_attribute_count  # 一个维度下最多允许多少个属性值,用来停止想LLM索取 属性值,控制 LLM 别生成太多
        self.infinite_threshold = infinite_threshold  # "无限节点"阈值：如果属性太多，是否合并处理
        self.tree_structure_file = tree_structure_file  # 保存树结构的文件路径
        # --- 运行时依赖 ---
        # 必须传入 LLM 引擎用于生成内容，传入线程池用于处理 CPU 密集型任务
        assert llm_engine is not None, "LLM engine must be provided."
        self.llm_engine = llm_engine
        assert threadpool_executor is not None, "Threadpool executor must be provided."
        self.threadpool_executor = threadpool_executor

        # 安全地获取 logger，如果 llm_engine 没有 logger 则为 None
        self.logger = getattr(self.llm_engine, "logger", None)

    def logging(self, msg, level="info"):
        """辅助日志函数，防止 self.logger 为 None 时报错"""
        if self.logger:
            getattr(self.logger, level)(msg)

    def is_leaf(self):
        """判断是否为叶子节点（没有子节点）"""
        return len(self.children) == 0

    def is_root(self):
        """判断是否为根节点"""
        return self.parent is None

    def is_infinite(self, attribute_values):
        """
            判断属性列表是否超过了"无限"阈值。
            如果是，系统可能会将所有属性打包到一个节点中处理，而不是每个属性创建一个子节点，
            以防止树的宽度爆炸。
        """
        return len(attribute_values) > self.infinite_threshold

    def __str__(self):
        """打印节点的简要信息"""
        dimension = self.dimension if self.dimension else "root"
        attribute_value = self.attribute_value if self.attribute_value else "None"

        return f"dimension: {dimension}\n" f"attribute_value: {attribute_value}"

    def to_dict(self):
        """
            [序列化] 将节点及其子树转换为字典格式。
            主要用于保存 checkpoint (JSON文件)，只保存数据状态，不保存运行时对象(LLM engine)。
            """
        return {
            "attribute_value": self.attribute_value,
            "dimension": self.dimension,
            "samples": self.samples,
            # 递归调用子节点的 to_dict
            "children": [child.to_dict() for child in self.children],
        }
    
    def count_infinite_nodes_in_path(self):
        """计算从当前点到根节点的路径上，有多少个节点被标记为列表/集合类型的属性（无限节点）"""
        count = 0
        current = self
        while current:#root的parent是None
            if isinstance(current.attribute_value, (list, set)):
                count += 1
            current = current.parent
        return count

    def retrieve_parents(self):
        """从叶子节点，回溯获取所有祖先节点（包括自己），顺序是从 子->父->根，方便后续 回溯 完整路径下的 划分准则和属性值"""
        parents = []
        current = self
        while current:
            parents.append(current)
            current = current.parent
        return parents

    def retrieve_dimension_values(self):
        """
                构建当前节点的完整路径的 准则 和 属性值。
                例如：[{"dimension": "Domain", "value": "Algebra"}, {"dimension": "Difficulty", "value": "Level 1"}]
                用于告诉 LLM 当前生成的是什么类型的数据。
                """
        parents = self.retrieve_parents()#返回包括自己在内的祖先节点 list
        parents = parents[:-1]# 去掉根节点（通常根节点没有具体的 dim/value）
        parents.reverse()# 反转顺序，变成 根（x）->父->子，符合人类阅读逻辑

        dimension_values = [] #完整路径 准则和属性值 list(dict,...)
        for parent in parents:
            dim = parent.dimension
            value = parent.attribute_value
            # 如果父节点的 value 是一个列表（Infinite Node），随机选一个值作为当前上下文
            # 这样做是为了让生成的样本具有多样性，而不是每次都包含所有属性
            if isinstance(value, (list, set)):
                value = random.choice(list(value))

            assert (dim is not None) and (
                value is not None
            ), "Dimension and attribute_value must not be None."
            dimension_values.append(
                {
                    "dimension": dim,
                    "attribute_value": value,
                }
            )
        return dimension_values

    def retrieve_parent_dimensions(self):
        """获取路径上已经用过的所有维度(划分准则)名称"""
        attribute_values = self.retrieve_dimension_values()
        dimensions = [d["dimension"] for d in attribute_values]

        return dimensions

    def retrieve_root(self):
        """找到树的根节点"""
        current = self
        while current.parent:
            current = current.parent
        return current

    def save_tree_structure(self, output_file):

        root = self.retrieve_root()

        pt = PrettyPrintTree(
            lambda x: x.children,
            lambda x: f"""dim: {x.dimension if x.dimension else "root"}\nattr: {x.attribute_value}\nchild_count:({len(x.children)})""",
            orientation=PrettyPrintTree.Horizontal# 水平方向打印
        )
        tree_as_str = pt(root, return_instead_of_print=True)
        # 去除 ANSI 颜色代码 (防止保存到 txt 文件乱码)
        ansi_escape = re.compile(r"(?:\x1B[@-_][0-?]*[ -/]*[@-~])")
        tree_as_str = ansi_escape.sub("", tree_as_str)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(tree_as_str)

        self.logging(
            f"Tree structure saved (full tree) to {output_file}.", level="info"
        )

    def format_dim_prompt(self):
        """generate prompt for selecting dimension and classifying"""
        """
            [Prompt Engineering] 生成用于"划分准则 和 对应分类属性"的提示词。
            任务：给 LLM 一堆题目，让它找出一个新的分类维度（Dimension），并将题目分类到该维度的属性（Attributes）中。
        """
        assert self.samples is not None, "Samples must be generated first."
        samples = ""
        for i, s in enumerate(self.samples, start=1):#用于在循环时同时获取索引和值，第二个参数是：指定索引的起始值
            samples += f"""{i}. {s}\n"""
        samples = samples.strip()#删去最后一个问题的 \n
        """
        1. 问题1
        2. 问题2
        3. 问题3
        4. .......
        """
        #检索完整路径上的 准则和属性值
        dimensions = self.retrieve_parent_dimensions()

        prompt = f"""
            As a math expert, your task is to examine the following MATH questions identify the SINGLE most significant dimension that characterizes the question space and differentiates these questions.
            Questions:
            {samples}
            
            Dimension Requirements:
            1. Core Dimension Identification: Identify exactly ONE core dimension that best distinguishes these questions.
            2. Excluded Dimensions: {', '.join(dimensions)}
            3. Unique Categorization: Each question MUST be categorized into exactly ONE attribute value.
            4. Mutually Exclusive Values: Attribute values must be mutually exclusive.
            5. Clarity in Values: Avoid ambiguous attribute values, such as "others".
            6. Independent Values: Each attribute must be a single distinct value - NO combined values like "attribute1_and_attribute2" or "attribute1/attribute2".
            
            Organize your responses in the following format without any extra text or explanations:
            {{
            "dimension": "dimension_name",
            "attributes": {{
                "attribute1": [list of sample indices],
                "attribute2": [list of sample indices],
                ...
            }}
            }}
            """ #在 Python 代码中使用 f"""..."""（格式化字符串，f-string）时，使用双重花括号 {{ 和 }} 是为了转义（Escape），从而在最终生成的字符串中保留字面意义上的单花括号。
        """
        prompt = 作为数学专家，你的任务是检查以下数学问题，识别出表征这些问题空间并区分这些问题的唯一最显著维度。
        问题列表： {samples}
        
        维度要求：
        1.核心维度识别：准确识别出一个最能区分这些问题的核心维度。
        2.排除的维度：{', '.join(dimensions)}
        3.唯一分类：每个问题必须被归类到准确的一个属性值中。
        4.互斥值：属性值必须是互斥的。
        5.数值清晰：避免使用模棱两可的属性值，例如 "others"（其他）。
        6.独立值：每个属性必须是一个单一且独特的数值——不能包含组合值，如 "attribute1_and_attribute2" 或 "attribute1/attribute2"。
        请按照以下格式组织你的回答，不要有任何额外的文本或解释： 
        { 
            "dimension": "维度名称", 
            "attributes": { 
            "属性1": [样本索引列表], 
            "属性2": [样本索引列表], ... 
            }
        }
        """
        return prompt


    def format_expand_prompt(self, dimension, attribute_values):
        """
                [Prompt Engineering] 生成用于"属性扩展"的提示词。
                任务：给定一个维度（如"难度"）和已知属性（如"简单"），让 LLM 脑暴出剩下的属性（如"中等", "困难"）。
        """
        prompt = f"""
        As a math expert, your task is to supplement the potential attribute values for a specified dimension in order to comprehensively model the entire space of questions.

        The whole space of questions is described as follows:
        1. Question Type: Questions must exclusively involve advanced mathematical reasoning suitable for competitions such as AMC 8, AMC 10, AMC 12, AIME, or USAMO.
        2. Difficulty Levels: Clearly assign each question a difficulty level from 1 (easy, typical of early AMC 8 questions) to 5 (very challenging, similar to later AIME/USAMO questions), consistent with recognized mathematical competition standards.
        3. Verb and Phrasing Diversity: Employ varied verbs and diverse phrasing, blending both clear interrogative questions and direct imperative instructions to maintain instruction diversity.
        4. Clarity and Uniqueness: Questions must provide all necessary details for a solver to determine exactly one unique solution without ambiguity.
        5. Notation and Formatting: Use clear and precise mathematical notation written in LATEX. If diagrams or illustrations are necessary, describe them explicitly in descriptive text or represent them using valid Asymptote code.
        6. Solvability: Questions should be solvable by advanced high school-level mathematical reasoning without calculators or external computational resources.
        
        Dimension: {dimension}
        Exiting attributes values: {json.dumps(attribute_values, indent=2)}#(要转换的Python对象, 缩进量：指定JSON字符串的格式化缩进, )
        
        Requirements for New Attribute Values:
        1. Clarity: Avoid ambiguous values, such as "others".
        2. Mutual Exclusivity: Ensure that attribute values do not overlap.
        3. Completeness: Ensure that all possible attribute values fully cover the dimension.
        4. Mathematical Complexity: Generate attribute values that reflect high school competition-level mathematical techniques and concepts.
        
        Organize your responses in the following format without any extra text or explanations:
        - If the existing attribute values completely cover the entire dimension, only output "null". For example,
        null
        - If the number of potential attribute values is more than 10, first output 10 potential new attribute values, and end your output with "infinite" in a new line. For example,
        attribute value 1
        attribute value 2
        ...
        attribute value 10
        infinite
        - Otherwise, output all the potential new attribute values, and end your output with "complete" in a new line. For example,
        attribute value 1
        attribute value 2
        ...
        attribute value n
        complete
        
        """

        """
        prompt = 
        作为数学专家，你的任务是为指定维度补充潜在的属性值，以便全面地对整个问题空间进行建模。

        整个问题空间的描述如下：
        1. 问题类型：问题必须仅涉及适合 AMC 8、AMC 10、AMC 12、AIME 或 USAMO 等竞赛的高级数学推理。
        2. 难度等级：根据公认的数学竞赛标准，清晰地为每个问题分配从 1（简单，典型的 AMC 8 早期问题）到 5（非常具有挑战性，类似于 AIME/USAMO 后期问题）的难度等级。
        3. 动词和措辞多样性：使用多样的动词和措辞，混合清晰的疑问句和直接的命令式指令，以保持指令的多样性。
        4. 清晰度和唯一性：问题必须提供所有必要的细节，以便解题者能够确定准确的一个唯一解，且没有歧义。
        5. 符号和格式：使用以 LaTeX 书写的清晰且精确的数学符号。如果需要图表或插图，请在描述性文本中明确描述它们，或使用有效的 Asymptote 代码表示它们。
        6. 可解性：问题应能够通过高中水平的高级数学推理解决，无需使用计算器或外部计算资源。
        
        维度：{dimension}
        现有属性值：{json.dumps(attribute_values, indent=2)}
        
        新属性值的要求：
        1. 清晰度：避免使用模棱两可的值，例如 "others"（其他）。
        2. 互斥性：确保属性值互不重叠。
        3. 完整性：确保所有可能的属性值完全覆盖该维度。
        4. 数学复杂性：生成的属性值要反映高中竞赛水平的数学技巧和概念。
        
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


        return prompt

    def format_gen_prompt(self):
        """
                [Prompt Engineering] 生成用于"样本生成"的提示词。
                任务：根据当前节点的路径上下文（如 Algebra -> Level 1），生成 10 个具体的数学题目。
        """
        if self.is_root():
            prompt = """
            As a math expert, generate 10 high school competition-level math questions in MATH dataset style.

            Each question should strictly adhere to these criteria:
            1. Question Type: Questions must exclusively involve advanced mathematical reasoning suitable for competitions such as AMC 8, AMC 10, AMC 12, AIME, or USAMO.
            2. Difficulty Levels: Clearly assign each question a difficulty level from 1 (easy, typical of early AMC 8 questions) to 5 (very challenging, similar to later AIME/USAMO questions), consistent with recognized mathematical competition standards.
            3. Verb and Phrasing Diversity: Employ varied verbs and diverse phrasing, blending both clear interrogative questions and direct imperative instructions to maintain instruction diversity.
            4. Clarity and Uniqueness: Questions must provide all necessary details for a solver to determine exactly one unique solution without ambiguity.
            5. Notation and Formatting: Use clear and precise mathematical notation written in LATEX. If diagrams or illustrations are necessary, describe them explicitly in descriptive text or represent them using valid Asymptote code.
            6. Solvability: Questions should be solvable by advanced high school-level mathematical reasoning without calculators or external computational resources.
            
            Organize your responses strictly in the following format without additional text or explanations:
            <Question 1>
                [Question text with proper mathematical notation]
            <Difficulty>
                [1-5]
            
            <Question 2>
                [Question text with proper mathematical notation]
            <Difficulty>
                [1-5]
            
            ...
            
            <Question 10>
                [Question text with proper mathematical notation]
            <Difficulty>
                [1-5]
            """

            """
            prompt = 
            作为数学专家，请以 MATH 数据集风格生成 10 道高中竞赛水平的数学问题。

            每个问题必须严格遵守以下标准：
            1. 问题类型：问题必须仅涉及适合 AMC 8、AMC 10、AMC 12、AIME 或 USAMO 等竞赛的高级数学推理。
            2. 难度等级：根据公认的数学竞赛标准，清晰地为每个问题分配从 1（简单，典型的 AMC 8 早期问题）到 5（非常具有挑战性，类似于 AIME/USAMO 后期问题）的难度等级。
            3. 动词和措辞多样性：使用多样的动词和措辞，混合清晰的疑问句和直接的命令式指令，以保持指令的多样性。
            4. 清晰度和唯一性：问题必须提供所有必要的细节，以便解题者能够确定准确的一个唯一解，且没有歧义。
            5. 符号和格式：使用以 LaTeX 书写的清晰且精确的数学符号。如果需要图表或插图，请在描述性文本中明确描述它们，或使用有效的 Asymptote 代码表示它们。
            6. 可解性：问题应能够通过高中水平的高级数学推理解决，无需使用计算器或外部计算资源。
            
            请严格按照以下格式组织你的回答，不要包含额外的文本或解释：
            <Question 1>
                [带有正确数学符号的问题文本]
            <Difficulty>
                [1-5]
            
            <Question 2>
                [带有正确数学符号的问题文本]
            <Difficulty>
                [1-5]
            
            ...
            
            <Question 10>
                [带有正确数学符号的问题文本]
            <Difficulty>
                [1-5]

            """

        else:
            attributes = self.retrieve_dimension_values()#获取路径上 准则 和 属性值 [{dis:.., attr:..}, ..]
            attributes_json = json.dumps(attributes, indent=2, ensure_ascii=False)# (要转换的Python对象, 缩进量：指定JSON字符串的格式化缩进, 非ASCII字符处理:False 表示不将非ASCII字符转换为Unicode转义序列|True（默认值）,中文字符会被转义)
            prompt = f"""
            As a math expert, generate 10 high school competition-level math questions in MATH dataset style.

            Each question should strictly adhere to these criteria:
            1. Question Type: Questions must exclusively involve advanced mathematical reasoning suitable for competitions such as AMC 8, AMC 10, AMC 12, AIME, or USAMO.
            2. Difficulty Levels: Clearly assign each question a difficulty level from 1 (easy, typical of early AMC 8 questions) to 5 (very challenging, similar to later AIME/USAMO questions), consistent with recognized mathematical competition standards.
            3. Verb and Phrasing Diversity: Employ varied verbs and diverse phrasing, blending both clear interrogative questions and direct imperative instructions to maintain instruction diversity.
            4. Clarity and Uniqueness: Questions must provide all necessary details for a solver to determine exactly one unique solution without ambiguity.
            5. Notation and Formatting: Use clear and precise mathematical notation written in LATEX. If diagrams or illustrations are necessary, describe them explicitly in descriptive text or represent them using valid Asymptote code.
            6. Solvability: Questions should be solvable by advanced high school-level mathematical reasoning without calculators or external computational resources.
            7.Attributes: Each question should be associated with all these attributes: {attributes_json}
            
            Organize your responses strictly in the following format without additional text or explanations:
            <Question 1>
                [Question text with proper mathematical notation]
            <Difficulty>
                [1-5]
            
            <Question 2>
                [Question text with proper mathematical notation]
            <Difficulty>
                [1-5]
            
            ...
            
            <Question 10>
                [Question text with proper mathematical notation]
            <Difficulty>
                [1-5]
            """

            """
            prompt = 
            作为数学专家，请以 MATH 数据集风格生成 10 道高中竞赛水平的数学问题。
            
            每个问题必须严格遵守以下标准：
            1. 问题类型：问题必须仅涉及适合 AMC 8、AMC 10、AMC 12、AIME 或 USAMO 等竞赛的高级数学推理。
            2. 难度等级：根据公认的数学竞赛标准，清晰地为每个问题分配从 1（简单，典型的 AMC 8 早期问题）到 5（非常具有挑战性，类似于 AIME/USAMO 后期问题）的难度等级。
            3. 动词和措辞多样性：使用多样的动词和措辞，混合清晰的疑问句和直接的命令式指令，以保持指令的多样性。
            4. 清晰度和唯一性：问题必须提供所有必要的细节，以便解题者能够确定准确的一个唯一解，且没有歧义。
            5. 符号和格式：使用以 LaTeX 书写的清晰且精确的数学符号。如果需要图表或插图，请在描述性文本中明确描述它们，或使用有效的 Asymptote 代码表示它们。
            6. 可解性：问题应能够通过高中水平的高级数学推理解决，无需使用计算器或外部计算资源。
            7. 属性：每个问题应与所有这些属性相关联：{attributes_json}
            
            请严格按照以下格式组织你的回答，不要包含额外的文本或解释：
            <Question 1>
                [带有正确数学符号的问题文本]
            <Difficulty>
                [1-5]
            
            <Question 2>
                [带有正确数学符号的问题文本]
            <Difficulty>
                [1-5]
            
            ...
            
            <Question 10>
                [带有正确数学符号的问题文本]
            <Difficulty>
                [1-5]
            
            """
        return prompt

    async def generate_samples_async(self):
        """
                [异步生成] 核心数据生成函数,生成枢轴样本
                异步生成样本数据，适用于需要并发处理或避免阻塞主线程的场景。
        """
        async def generate_subsamples():
            """ 定义内部异步函数，用于封装实际的生成逻辑"""
            single_prompt = self.format_gen_prompt()#获取生成枢轴样本的prompt
            #调用 LLM 引擎生成文本
            responses = await self.llm_engine.generate_batch_async([single_prompt])

            all_samples = []
            # 使用正则表达式解析 XML 风格的标签 <Question N>...</Question N>
            # 模式解释：
            # - <Question\s+(\d+)> 匹配 <Question N> 标签，N 被捕获为组1
            # - \s*(.*?) 匹配标签后的内容（非贪婪匹配）
            # - (?=\s*<Question\s+\d+>|$) 正向预查：匹配到下一个<Question>标签或字符串结尾
            # - re.DOTALL 标志使 . 匹配包括换行符在内的所有字符
            pattern = r"<Question\s+(\d+)>\s*(.*?)(?=\s*<Question\s+\d+>|$)"

            for idx, raw_text in enumerate(responses, start=1):
                self.logging(f"[Prompt {idx}] raw response = {raw_text}", level="debug")
                # 使用正则表达式从原始文本中提取所有匹配的样本
                # re.findall 返回所有匹配的列表，每个匹配是 (问题编号, 问题文本) 的元组
                matches = re.findall(pattern, raw_text, flags=re.DOTALL)
                for qnum, qtext in matches:
                    all_samples.append(qtext.strip())# 清理文本：去除首尾空白字符，并添加到样本列表
            return all_samples


        all_samples = await generate_subsamples()
        self.samples = all_samples
        return self.samples

    def select_dimension_and_classify(self):
        """
                [同步逻辑] 调用 LLM 识别维度并分类样本。
                注意：实际主要使用下方的 async 版本，这个可能是保留的旧代码或用于调试。
        """
        parent_dimensions = self.retrieve_parent_dimensions()#获取完整路径划分准则

        prompt = self.format_dim_prompt()#获取提示词,其中逻辑已经获取完整路径划分准则和对应属性
        response = self.llm_engine.generate_batch(prompt, max_tokens=1024, temperature=1.0)#获取LLM的回答
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


    
    async def select_dimension_and_classify_async(self, max_attempts=5):
        """
                [异步分类] 让 LLM 决定如何分裂当前节点, 即 选择 划分准则 和 属性
                包含  重试机制，直到 LLM 返回合法的 JSON 且逻辑正确（所有样本都被分类，且没有遗漏）。
        """
        for attempt in range(max_attempts):
            parent_dimensions = self.retrieve_parent_dimensions()#完整路径的 划分准则
            prompt = self.format_dim_prompt()#生成 划分准则 和 属性 的提示词

            # 调用 LLM，temperature=1.0 鼓励多样性
            responses = await self.llm_engine.generate_batch_async(
                prompt, max_tokens=1024, temperature=1.0
            )
            response = responses[0] if isinstance(responses, list) else responses
            # 解析 LLM 返回的 JSON
            candidates = parse_json_candidates(response, logger=self.logger, debug=True)#返回列表
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
                f"[Attempt {attempt+1}/{max_attempts}] Parsed dimension candidates: {candidates}",
                level="debug"
            )

            found_dim = None
            for c in candidates:
                valid = True
                if isinstance(c, dict) and ("dimension" in c) and ("attributes" in c):# 校验 JSON 结构：必须包含 dimension 和 attribut，过滤掉包含进来的[样本索引列表]
                    dim = c["dimension"].strip()
                    attr_map = c["attributes"]#属性值的字典

                    all_indices = set()
                    for cat, arr in attr_map.items():#获取每一个属性值，items()：返回字典的(键, 值)对
                        if not isinstance(arr, list) or any(
                            not (isinstance(x, int) and 1 <= x <= len(self.samples))
                            for x in arr
                        ):# 1) arr不是列表，或者 2) arr中有无效元素
                            valid = False
                            self.logging(f"Invalid attribute '{cat}': {arr}.", level="debug")
                            break
                        all_indices.update(arr)#将可迭代对象中的所有元素添加到集合中

                    if valid:# 校验逻辑完整性：所有样本是否都被覆盖？维度名是否重复？
                        if all_indices == set(range(1, len(self.samples) + 1)) \
                        and dim not in parent_dimensions:
                            found_dim = c
                            break

            if found_dim is not None:
                return found_dim# 成功找到有效 划分准则和属性
            else:
                self.logging(
                    f"No valid dimension classification found in attempt {attempt+1}, will retry...",
                    level="warning"
                )

        self.logging(
            f"Failed to classify dimension after {max_attempts} attempts => skip node expansion.",
            level="error"
        )
        return None


    async def expand_dimension_async(self, dimension, attribute_values):
        """
                [异步扩展] 补全维度的其他可能属性值。
                循环调用，直到 LLM 说 "complete" (完成) 或 "null" (不需要扩展)。
        """
        attempts = 0
        while True:
            prompt = self.format_expand_prompt(dimension, attribute_values)
            responses = await self.llm_engine.generate_batch_async(prompt, max_tokens=1024, temperature=0.7)
            raw_res = responses[0] if isinstance(responses, list) else responses
            # 解析每一行属性
            candidates = parse_expand_attributes_from_str(raw_res, logger=self.logger, debug=True)
            if (not candidates) or (candidates[-1] not in ["null", "infinite", "complete"]):
                attempts += 1
                self.logging(f"Attempt {attempts}: invalid response => retrying", level="warning")
                continue# 格式错误，重试

            elif candidates[-1] == "infinite": #todo 这里已经用到挤牙膏的思想了，以实现更加多样性
                # LLM 认为属性还有很多，先加进去，然后继续循环生成
                # candidates[:-1] 的意思是：取列表里除了最后一个元素（"infinite"）之外的所有东西。
                # 这一步把 LLM 这次吐出来的 10 个新属性，追加到总列表 attribute_values 里。
                attribute_values += candidates[:-1]
                # 2. 安全熔断机制 (Safety Brake)
                # 这一步非常关键。如果没有这一步，如果 max_attribute_count 设得很大，
                # 或者 LLM 一直傻乎乎地回 infinite，程序就会陷入死循环，直到内存爆炸。
                if len(attribute_values) > self.max_attribute_count:
                    break# 强制截断，防止无限循环
                else:
                    self.logging("Insufficient attributes for infinite => continue refilling", level="warning")
                    continue# 继续生成下一批
            elif candidates[-1] == "null":
                # LLM 认为现有的属性已经够了
                self.logging(
                    f"No valid expansion info found for dimension '{dimension}', use original attributes.",
                    level="warning",
                )
                break
            elif candidates[-1] == "complete":
                # LLM 补充了一些属性，并表示结束了
                attribute_values += candidates[:-1]
                break
            else:
                self.logging(f"Unexpected last candidate: {candidates[-1]}", level="warning")
                continue

        return attribute_values#当前节点扩展后的属性列表

    """主程序入口"""
    async def expand_nodes_async(self, output_file=None, result_file=None):
        """
        [BFS 主循环] 广度优先搜索扩展整棵树。
        这是整个生成过程的入口函数。
        """
        # 1. 保存初始树结构
        self.save_tree_structure(self.tree_structure_file)
        # 2. 扩展当前节点（通常是 Root），获取第一层子节点
        children = await self._expand_single_node_async(output_file, result_file)
        # 3. 初始化 BFS 队列
        queue = deque(children)

        level = 0
        while queue:
            level_size = len(queue)
            tasks = []
            self.logging(f"[BFS] Start processing level={level} with {level_size} nodes", level="info")
            # 4. 批量处理当前层的所有节点
            for _ in range(level_size):
                node = queue.popleft()
                # 为每个节点创建一个异步任务，并发执行 _expand_single_node_async，放入执行队列中 ”排号“
                tasks.append(asyncio.create_task(
                    node._expand_single_node_async(output_file, result_file)
                ))
            # 5. 等待当前层所有节点处理完成
            results = await asyncio.gather(*tasks)#主程序暂停并让出CPU去执行 队列中的后台任务

            # 6. 收集下一层的子节点加入队列
            # BFS: collect next level
            for child_list in results:
                for c in child_list:
                    queue.append(c)
            # 每层处理完保存一次结构
            self.save_tree_structure(self.tree_structure_file)
            level += 1
        # 最终保存
        self.save_tree_structure(self.tree_structure_file)

    async def _expand_single_node_async(self, output_file, result_file):
        """
        [单节点逻辑] 处理一个节点的具体扩展逻辑。
        分为三种情况：根节点、达到最大深度的节点、普通中间节点。

            [单节点逻辑] 处理一个节点的具体扩展逻辑。

            逻辑流：
            1. 如果是根节点 -> 创建 7 个固定的数学领域子节点。
            2. 如果达到最大深度 -> 停止分裂，根据路径上的"无限节点"数量计算采样量，生成最终数据。
            3. 如果是中间节点 ->
               a. 先生成少量样本让 LLM 找规律 (Dimension)。
               b. 扩展该 Dimension 下的所有可能属性 (Attributes)。
               c. 判断属性数量是否过多：
                  - 过多 (Infinite) -> 创建 1 个聚合子节点 (防止宽度爆炸)。
                  - 正常 -> 创建 N 个普通子节点。

        """
        config = getattr(self.llm_engine, "config", {})
        # 获取全局配置，默认无限路径（包含 无限节点的路径）采样倍率为 3
        infinite_path_samples = config.get("infinite_path_samples", 3)

        # =================================================================
        # Case 1: 根节点处理 (Root Node)
        # 根节点不依赖 LLM 动态生成，而是硬编码了 MATH 数据集的 7 大核心领域。
        # 这样保证了树的起始分类是标准且覆盖全面的。
        # =================================================================
        if self.is_root():
            self.logging("Root node detected. Creating predefined MATH domain children.", level="info")
            
            dimension = "Mathematical Domain"
            attribute_values = [
                "Algebra", 
                "Counting & Probability", 
                "Geometry", 
                "Intermediate Algebra", 
                "Number Theory", 
                "Prealgebra",
                "Precalculus"
            ]
            
            self.children = []
            for attr in attribute_values:
                # 创建下一层子节点对象
                child = type(self)(  #创建一个与当前实例同类型的新对象
                    depth=self.depth + 1,
                    llm_engine=self.llm_engine,
                    dimension=dimension,
                    attribute_value=attr,
                    parent=self,
                    max_depth=self.max_depth,
                    num_samples_per_node=self.num_samples_per_node,
                    infinite_threshold=self.infinite_threshold,
                    max_attribute_count=self.max_attribute_count,
                    threadpool_executor=self.threadpool_executor,
                    tree_structure_file=self.tree_structure_file,
                )
                self.children.append(child)
            # [Checkpoint] 保存树结构到 JSON 文件，防止程序中断丢失进度
            if output_file:
                tree_dict = self.retrieve_root().to_dict()
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(tree_dict, f, ensure_ascii=False, indent=2)
            # 返回子节点列表，供 BFS 队列继续处理
            return self.children
        # =================================================================
        # Case 2: 达到最大深度 (Max Depth Reached) -> 变为叶子节点   =子空间数据合成==
        # 树不再长高了，现在任务转变为：根据当前路径的约束，生成具体的 问题 数据。
        # =================================================================
        if self.depth >= self.max_depth:
            self.logging(f"[Leaf@MaxDepth] depth={self.depth}, stop expansion.", level="info")
            # --- 动态采样量计算 (关键逻辑) ---
            # 如果从根节点到这里的路径上经过了"无限节点" (aggregated node)，
            # 说明这个分类路径涵盖的范围非常广（例如包含了"所有整数"）。
            # 为了保证数据覆盖率，经过的无限节点越多，这里生成的样本量就要指数级增加。
            # 公式：total_samples = base ^ infinite_count
            infinite_count = self.count_infinite_nodes_in_path()
            total_samples = max(1, infinite_path_samples ** infinite_count)#如果没有无限节点，它是“生成 1 组默认数量（prompt中 明确指明10个）的题目”,是由prompt的中的内容决定的
            self.logging(f"Path has {infinite_count} infinite nodes, generating {total_samples} sample sets", level="info")

            # 循环生成多批次数据
            all_samples = []
            for i in range(total_samples):
                samples = await self.generate_samples_async()
                all_samples.extend(samples)
                if i > 0:
                    self.logging(f"Generated sample set {i+1}/{total_samples} for infinite path", level="info")
            
            self.samples = all_samples
            # [I/O] 将生成的具体题目追加写入结果文件 (.jsonl 格式)
            if result_file:
                with open(result_file, "a", encoding="utf-8") as f:
                    for q in all_samples:
                        line = {"question": q}
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
            # [Checkpoint] 保存树结构（虽然没有新子节点，但 samples 属性更新了）
            if output_file:
                tree_dict = self.retrieve_root().to_dict()
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(tree_dict, f, ensure_ascii=False, indent=2)
            return []# 返回空列表，表示没有下一层了

        # =================================================================
        # Case 3: 普通中间节点 (Intermediate Node) -> 尝试分裂
        # 这里的核心任务是：找出下一个细分维度 (Dimension)，以及对应的分类属性
        # =================================================================

        # 3.1 生成"种子"样本
        # 先生成少量10个枢轴样本
        samples = await self.generate_samples_async()# 返回的是 self.samples,已经将生成的内容传给self.samples,select_dimension_and_classify_async中的获取prompt可以在self.samples中获取当前节点的 枢轴样本
        dim_dict = await self.select_dimension_and_classify_async(max_attempts=5)
        # 如果尝试多次 LLM 还是无法分类（比如样本太杂乱），则放弃分裂，将当前节点降级为叶子节点。
        if dim_dict is None:
            self.logging("Dimension classification failed => treat this node as leaf.", "warning")

            # todo 自己修改 执行无限节点步长
            infinite_count = self.count_infinite_nodes_in_path()
            # 计算还需要生成多少轮 (总轮数 - 已经生成的1轮)
            total_samples = max(0, infinite_path_samples ** infinite_count - 1)
            final_samples = list(samples)  # 拷贝一份，避免污染引用

            if total_samples > 0:
                for i in range(total_samples):
                    # 注意：这里 generate_samples_async 依然会覆盖 self.samples，
                    # 但我们在最后会手动修正它。
                    extra_samples = await self.generate_samples_async()
                    final_samples.extend(extra_samples)
                    self.logging(f"Generated additional batch {i + 1}/{total_samples}", level="info")

            # todo 自己修改手动将 self.samples 更新为完整的列表
            self.samples = final_samples

            """============================================================================"""

            if result_file:#直接将生成的枢轴样本 作为当前叶子节点生成的 最终问题样本
                with open(result_file, "a", encoding="utf-8") as f:
                    for q in samples:
                        line = {"question": q}
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
            return []# 返回空列表，表示没有下一层了
        # 3.3 属性扩展 (Attribute Expansion)
        dimension = dim_dict["dimension"]
        attribute_list = list(dim_dict["attributes"].keys())
        expanded_list = await self.expand_dimension_async(dimension, attribute_list)
        # 将扩展出来的新属性加入字典（目前它们还没有对应的样本，是空的）
        for attr in expanded_list:
            if attr not in dim_dict["attributes"]:
                dim_dict["attributes"][attr] = []
        # 获取最终的所有属性列表
        all_attrs = list(dim_dict["attributes"].keys())
        # =================================================================
        # 3.4 拓扑结构决策 (Topology Decision) - 重点！
        # 决定是"水平平铺"还是"垂直聚合"。
        # =================================================================
        if self.is_infinite(all_attrs):
            child = type(self)(
                depth=self.depth + 1,
                llm_engine=self.llm_engine,
                dimension=dimension,
                attribute_value=all_attrs,#无限节点的 属性值 变成列表
                parent=self,
                max_depth=self.max_depth,
                num_samples_per_node=self.num_samples_per_node,
                infinite_threshold=self.infinite_threshold,
                max_attribute_count=self.max_attribute_count,
                threadpool_executor=self.threadpool_executor,
                tree_structure_file=self.tree_structure_file,
            )
            self.children = [child]
        else:
            # [分支 B: 正常分裂]
            # 如果属性数量正常，为每个属性创建一个独立的子节点。
            self.children = []
            for attr in all_attrs:
                c = type(self)(
                    depth=self.depth + 1,
                    llm_engine=self.llm_engine,
                    dimension=dimension,
                    attribute_value=attr,
                    parent=self,
                    max_depth=self.max_depth,
                    num_samples_per_node=self.num_samples_per_node,
                    infinite_threshold=self.infinite_threshold,
                    max_attribute_count=self.max_attribute_count,
                    threadpool_executor=self.threadpool_executor,
                    tree_structure_file=self.tree_structure_file,
                )
                self.children.append(c)
        # =================================================================
        # 3.5 边缘情况兜底 (Edge Case Fallback)
        # 如果因为某种奇怪原因（如属性列表为空）没有创建出任何子节点。
        # =================================================================
        if not self.children:
            self.logging(f"[Leaf] Node dimension={dimension} => leaf node with no children, writing samples.", level="info")
            # 同样需要执行"无限路径补偿"逻辑，因为这本质上也是一个叶子节点
            # 注意这里是 total_samples - 1，因为前面已经在 Step 3.1 生成过 1 批 samples 了
            infinite_count = self.count_infinite_nodes_in_path()
            total_samples = max(0, infinite_path_samples ** infinite_count - 1)

            #todo 自己修改
            final_samples = list(samples)
            
            if total_samples > 0:
                additional_samples = []
                for i in range(total_samples):
                    extra_samples = await self.generate_samples_async()
                    additional_samples.extend(extra_samples)
                    self.logging(f"Generated additional sample set {i+1}/{total_samples} for infinite path (leaf node)", level="info")
                samples.extend(additional_samples)

            # todo 自己修改手动将 self.samples 更新为完整的列表
            self.samples = final_samples
            
            if result_file:
                with open(result_file, "a", encoding="utf-8") as f:
                    for q in samples:
                        line = {"question": q}
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
        # 写入结果
        if output_file:
            tree_dict = self.retrieve_root().to_dict()
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(tree_dict, f, ensure_ascii=False, indent=2)

        return self.children



    @staticmethod
    def from_dict(d: dict, parent=None):
        """
        作用是将存储在硬盘上的静态字典数据（通常来自 JSON 文件），重新构建成内存中活跃的 TreeNode 对象树。这是实现断点续训 (Checkpoint Loading) 的关键步骤。
        [静态工厂方法] 从字典反序列化还原 TreeNode 对象。

        该方法通常用于从 JSON 文件中加载之前的生成进度。
        它是一个递归函数：先恢复自己，再让孩子们恢复自己。

        Args:
            d (dict): 包含节点数据的字典 (通常由 to_dict 生成)。
            parent (TreeNode, optional): 父节点对象。在递归调用时会自动传入。

        Returns:
            TreeNode: 重建完成的节点对象 (及其整棵子树)。
        """

        # 1. [创建节点外壳]
        # 使用字典里的数据初始化一个新的 TreeNode 对象。
        # 注意：这里有几个关键点：
        #   a. llm_engine 和 threadpool_executor 设为 None。
        #      因为 JSON 文件只存数据，存不了活跃的数据库连接或线程池。
        #      这些"运行时依赖"需要在整棵树重建好之后，通过 inject_runtime_to_tree 函数注入。
        #   b. depth 暂时设为 0，后面会根据 parent 修正。
        #   c. 配置参数 (max_depth 等) 这里使用了硬编码的默认值。
        #      (改进建议：最好也能从 d 字典里读取这些配置，保证配置一致性)
        node = TreeNode(
            depth=0,
            llm_engine=None,  # <--- 缺失的灵魂 (运行时组件)
            threadpool_executor=None,  # <--- 缺失的动力 (线程池)
            dimension=d.get("dimension"),  # 恢复维度名称
            attribute_value=d.get("attribute_value"),  # 恢复属性值
            max_depth=5,
            num_samples_per_node=10,
            infinite_threshold=50,
            max_attribute_count=50,
            parent=parent,  # 建立父子引用 (指向父节点)
            tree_structure_file="tree_structure.txt",
        )
        # 2. [恢复数据载荷]
        # 将之前生成的题目样本列表 (Samples) 塞回节点里。
        node.samples = d.get("samples", [])
        # 3. [递归重建子树 - 核心逻辑]
        # d.get("children", []) 获取的是一个包含子节点字典的列表。
        # 我们遍历这个列表，对每一个子字典再次调用 TreeNode.from_dict。
        # 这种"自己调用自己"的过程会一直持续到叶子节点，从而重建整棵树的拓扑结构。
        for child_dict in d.get("children", []):
            # 关键：传入当前的 node 作为 child 的 parent
            child_node = TreeNode.from_dict(child_dict, parent=node)
            # 建立父子引用 (父节点指向子节点)
            node.children.append(child_node)
        # 4. [修正深度]
        # 如果当前节点有父节点，那么它的深度应该是 父节点深度 + 1。
        # 如果是根节点 (parent is None)，depth 保持为 0。
        if parent:
            node.depth = parent.depth + 1
        # 5. [返回重生对象] - 完整的树
        return node

def inject_runtime_to_tree(node: TreeNode,
                           llm_engine,
                           threadpool_executor,
                           max_depth,
                           infinite_threshold,
                           max_attribute_count):
    """当你从 JSON 文件通过 from_dict 恢复了一棵树后，这棵树只是一个**“植物人”**（只有数据，没有行动能力）。这个函数负责把 大脑 (LLM Engine)、动力 (ThreadPool) 和 规则 (Config) 重新注入到每一个节点中，让树重新“活”过来，继续生长

        [依赖注入函数] 递归地将运行时组件和全局配置注入到整棵树的每一个节点中。

        通常配合 TreeNode.from_dict() 使用：
        1. from_dict() 负责恢复静态数据结构 (拓扑结构、题目内容)。
        2. inject_runtime_to_tree() 负责恢复动态行为能力 (API 调用能力、并发计算能力)。

        Args:
            node (TreeNode): 当前需要注入的树节点对象。
            llm_engine (LLMInference): 大模型推理引擎实例，赋予节点生成文本的能力。
            threadpool_executor (ThreadPoolExecutor): 线程池执行器，赋予节点并发处理 CPU 任务的能力。
            max_depth (int): 全局配置 - 树的最大生长深度。
            infinite_threshold (int): 全局配置 - 判定“无限节点”的阈值。
            max_attribute_count (int): 全局配置 - 一个维度下允许的最大属性数量。

        Returns:
            None: 此函数为原地操作 (In-place operation)，直接修改传入的 node 对象，不返回新对象。
    """
    # 1. [注入运行时依赖]
    # 将 LLM 引擎实例赋值给当前节点。
    # 没有这个，节点调用 self.generate_samples_async() 时会报错。
    node.llm_engine = llm_engine
    # 将线程池赋值给当前节点。
    # 没有这个，节点调用 asyncio.to_thread() 时无法利用预设的线程资源。
    node.threadpool_executor = threadpool_executor
    # 2. [注入全局配置]
    # 更新或重置该节点的配置参数。
    # 这允许我们在从断点恢复时，动态调整配置（例如想让树长得更深，可以在这里修改 max_depth）
    node.max_depth = max_depth
    node.infinite_threshold = infinite_threshold
    node.max_attribute_count = max_attribute_count
    # 3. [递归处理子节点]
    # 遍历当前节点的所有孩子节点
    for child in node.children:
        # [修复父子引用]
        # 在 JSON 反序列化时，parent 引用可能会丢失或指向错误。
        # 这里强制将子节点的 parent 指向当前节点 (self)，确保双向链表结构 (Parent <-> Child) 的完整性。
        child.parent = node
        # [校准深度]
        # 强制将子节点的深度设为当前深度 + 1，防止数据记录错误。
        child.depth = node.depth + 1
        # [递归调用 - DFS]
        # 这样会一直传递到叶子节点，确保整棵树所有角落都被“激活”。
        inject_runtime_to_tree(child,
                               llm_engine,
                               threadpool_executor,
                               max_depth,
                               infinite_threshold,
                               max_attribute_count)

class TeeToFile:
    def __init__(self, original_stream, file_path, mode="w", encoding="utf-8"):
        self.original_stream = original_stream
        self.file = open(file_path, mode=mode, encoding=encoding, buffering=1)

    def write(self, data):
        self.original_stream.write(data)
        self.original_stream.flush()
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.original_stream.flush()
        self.file.flush()

    def isatty(self):
        return True

    def close(self):
        self.file.close()


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

