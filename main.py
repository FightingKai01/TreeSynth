import os
import json
import logging
import datetime
import signal
import sys
import concurrent.futures
import asyncio
import shutil
import random
import torch
import numpy as np
from config import DEFAULT_CONFIG, BACKEND_CONFIGS, VLLM_API_POOL, OPENAI_API_POOL
from mcts_generator_gsm_async import LLMInference, APIPool, TreeNode, inject_runtime_to_tree

def set_seed(seed: int = 42):
    """
        设置全局随机种子以确保实验结果的可复现性。

        参数:
            seed (int): 随机种子数值，默认为 42。
    """
    # 1. 设置 Python 哈希种子
    # 禁止 Python 对字符串哈希的随机化，确保字典(dict)和集合(set)等哈希结构的迭代顺序在不同运行中保持一致。
    # 必须在任何 Python 进程启动前或脚本最开始设置才有效。
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 2. 设置 Python 内置 random 模块的种子
    # 影响所有使用 random 模块的操作 (如 random.choice, random.shuffle)
    random.seed(seed)
    # 3. 设置 NumPy 的随机种子
    # 影响所有 numpy.random 相关的操作 (如数据预处理、增强中常用的 np.random.rand 等)
    np.random.seed(seed)
    # 4. 设置 PyTorch CPU 的随机种子
    # 影响 PyTorch 在 CPU 上进行的随机操作 (如权重初始化)。
    torch.manual_seed(seed)
    # 5. 设置 PyTorch 当前 GPU 的随机种子
    # 影响当前选定的 GPU 上的随机操作。
    torch.cuda.manual_seed(seed)
    # 6. 设置 PyTorch 所有可用 GPU 的随机种子
    # 如果使用多卡训练 (Multi-GPU)，这行代码确保所有卡都设置了相同的种子。
    torch.cuda.manual_seed_all(seed)
    # 7. 强制 cuDNN 使用确定性算法
    # cuDNN 库中某些卷积算法在不同运行间可能产生微小的数值差异。
    # 设置为 True 会强制仅使用确定性的卷积算法，但这可能会导致计算速度下降。
    torch.backends.cudnn.deterministic = True
    # 8. 关闭 cuDNN 的 Benchmark 模式
    # 当设置为 True 时，cuDNN 会在开始时尝试多种算法来寻找针对当前硬件最快的卷积实现。
    # 由于硬件状态波动，每次选择的算法可能不同，从而导致结果不一致。
    # 关闭它 (False) 可以确保每次使用相同的算法，增强复现性。
    torch.backends.cudnn.benchmark = False

def signal_handler(signum, frame):
    logger = logging.getLogger()
    logger.info(f"Received signal {signum}, terminating.")
    sys.exit(1)

def setup_logger(log_file="generation.log"):
    """
        配置并返回一个全局 logger 对象。
        功能：同时将日志输出到控制台（屏幕）和指定的文件中。

        参数:
            log_file (str): 日志文件的保存路径，默认为 "generation.log"。
    """
    # 1. 获取根日志记录器 (Root Logger)
    # logging.getLogger() 如果不传参数，获取的是全局唯一的根记录器。
    # 这意味着你在程序的任何其他地方调用 logging.info() 都会受到这个配置的影响。
    logger = logging.getLogger()
    # 2. 设置全局日志级别
    # logging.DEBUG 是最低级别，意味着允许所有级别的日志（Debug, Info, Warning, Error, Critical）通过。
    # 如果这里设置得太高（如 WARNING），即使后面的 Handler 设置了 DEBUG，低级别日志也会在这一步被拦截。
    logger.setLevel(logging.DEBUG)
    # 3. 清除旧的 Handlers (关键步骤)
    # logger.hasHandlers() 检查当前 logger 是否已经绑定了处理器。
    # 如果已有 Handler（例如在 Jupyter Notebook 中重复运行该单元格，或者多次调用此函数），
    # 必须先清除它们，否则会导致同一条日志被打印多次（重复输出）。
    if logger.hasHandlers():
        logger.handlers.clear()
    # 4. 创建控制台处理器 (StreamHandler)
    # 用于将日志输出到标准输出流（即屏幕/终端）。
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # 5. 创建文件处理器 (FileHandler)
    # 用于将日志写入文件。
    # mode="w": 写模式。每次运行都会清空旧文件，重新开始写入。
    #           如果希望保留旧日志追加写入，请改为 mode="a" (append)。
    # encoding="utf-8": 强制使用 UTF-8 编码，防止中文日志在 Windows 上出现乱码。
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    # 6. 定义日志格式 (Formatter)
    # %(asctime)s:  日志发生的时间，精确到毫秒。
    # %(levelname)s: 日志级别名称 (INFO, ERROR, DEBUG 等)。
    # %(message)s:   具体的日志内容。
    # 示例输出: 2025-01-06 12:00:00,123 - INFO - 任务开始执行
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # 7. 将格式应用到处理器
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # 8. 将处理器添加到 logger
    # 只有添加了 Handler，logger 才知道要把日志发往哪里。
    logger.addHandler(ch)
    logger.addHandler(fh)
    # 9. 返回配置好的 logger 对象供外部使用
    return logger

async def run_generation_process(
    log_file,
    output_file,
    tree_structure_file,
    result_file,
    backend="vllm",
    use_api_pool=False,
    config_override=None
):
    """
        执行数据生成的异步主流程。
        负责初始化环境、加载之前的进度（如果有）、调度 LLM 推理并保存结果。

        async def 和 await 是 Python 异步编程（Asyncio）的两个核心关键字。它们组合在一起构成了一种**“非阻塞式”**的任务处理机制。
        async def	声明者	“我是一个异步任务，我内部可能有需要等待的操作。”
        await	调度员	“我现在开始等了，CPU 你先去忙别的，等结果回来了叫我。”
    """
    # 1. 获取日志记录器
    # 获取根 Logger 实例，用于记录整个流程的状态信息。
    logger = logging.getLogger()#可以获取到全局的logger
    logger.info(f"Starting data generation process with backend: {backend}")
    # 2. 配置初始化与合并（左右相关的 配置参数，均合并到config中）
    # 采用三层配置优先级策略：默认配置 < 后端特定配置 < 用户手动覆盖配置。

    # 第一层：加载全局默认配置（防止修改原字典，使用 copy）。
    config = DEFAULT_CONFIG.copy()
    # 第二层：如果指定的 backend (如 "vllm" 或 "openai") 有特定配置，进行更新。
    if backend in BACKEND_CONFIGS:
        config.update(BACKEND_CONFIGS[backend])
    # 第三层：如果有传入特定的覆盖参数（config_override），优先级最高，覆盖之前的设置。
    if config_override:
        config.update(config_override)
    # 3. 初始化 API 连接池 (可选)
    # 如果启用 API 池 (use_api_pool=True)，则根据后端类型加载对应的 API Key/Endpoint 列表。
    # 这通常用于轮询多个 API Key 以提高并发限额。
    api_pool = None
    if use_api_pool:
        if backend == "vllm":
            logger.info("Using vLLM API pool")
            api_pool = APIPool(VLLM_API_POOL)
        elif backend == "openai":
            logger.info("Using OpenAI API pool")
            api_pool = APIPool(OPENAI_API_POOL)
    # 4. 初始化 LLM 推理引擎
    # LLMInference 是对底层模型调用的封装类。
    # 将配置参数（重试次数、最大工作线程数、并发限制）解包传入。
    llm_engine = LLMInference(
        backend=backend,
        api_pool=api_pool,
        config=config,
        logger=logger,
        max_retries=config.get("max_retries", 5),
        max_workers=config.get("max_workers", 64),
        max_concurrent_requests=config.get("max_concurrent_requests", 64)
    )
    # 5. 注册信号处理器 (Graceful Shutdown)
    # 捕获 SIGINT (Ctrl+C) 和 SIGTERM (终止信号)。
    # 确保在用户强制停止脚本时，程序有机会执行保存数据、关闭连接等收尾工作，而不是直接崩溃。
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"Initializing {backend} Inference Engine...")
    # 6. 创建线程池执行器 (ThreadPoolExecutor)
    # 虽然 Python 的 async/await 适合处理 I/O，但在处理某些 CPU 密集型任务（如解析复杂 JSON、计算指标）时，
    # 仍然需要线程池来避免阻塞主事件循环 (Event Loop)。"""async/await 只能解决“等”的问题，解决不了“算”的问题。API 等待时：await 就像是把任务挂在墙上，CPU（调度员）去干别的了，这没问题。计算复杂时：一旦 API 回复了，CPU 必须亲自回来处理这些数据。如果这个处理过程（比如大规模数据清洗、复杂的模型后处理、甚至是一个死循环）需要耗时 5 秒，那么在这 5 秒内，整个事件循环是卡死的。"""
    threadpool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.get("max_workers", 64))

    # 7. 断点续训逻辑 (Checkpoint Loading)
    # 尝试从 output_file 中恢复之前的生成树，避免每次都从头开始。
    root_node = None
    if os.path.exists(output_file):
        logger.info(f"Detected existing {output_file}, attempting to load previous tree...")
        try:
            # 读取旧的 JSON 数据
            with open(output_file, "r", encoding="utf-8") as f:
                tree_data = json.load(f)
            # 将 JSON 字典反序列化为 TreeNode 对象结构
            root_node = TreeNode.from_dict(tree_data, parent=None)
            # 关键步骤：注入运行时依赖 (Dependency Injection)
            # 因为 JSON 文件只存储了数据（静态属性），没有存储运行时对象（LLM 引擎、线程池等）。
            # 这里必须手动将当前的 engine 和 executor 注入到恢复出的节点树中，树才能继续“生长”。
            inject_runtime_to_tree(
                root_node,
                llm_engine,
                threadpool_executor,
                config.get("max_depth", 6),
                config.get("max_sample_infinite_attribute", 25),
                config.get("max_attribute_count", 25)
            )
            logger.info("Successfully loaded existing tree. Will continue expansion.")
        except Exception as e:
            # 如果加载失败（如文件损坏），记录错误并降级为“重新开始”模式
            logger.error(f"Failed to load {output_file} due to {e}. Will start fresh.")
            root_node = None
    # 8. 初始化新树 (如果无需/无法恢复)
    # 如果没有旧文件或加载失败，则创建一个全新的根节点 (Root Node)。
    if root_node is None:
        root_node = TreeNode(
            llm_engine=llm_engine,#本身具备一个线程池
            threadpool_executor=threadpool_executor,#树节点类也需要一个线程池
            tree_structure_file=tree_structure_file,#输出的可视化文件
            depth=0,
            parent=None,
            dimension=None,  
            attribute_value=None,
            max_depth=config.get("max_depth", 4),
            num_samples_per_node=config.get("num_samples_per_node", 10),#代表样本数量
            infinite_threshold=config.get("max_sample_infinite_attribute", 25),# 无限属性的阈值
            max_attribute_count=config.get("max_attribute_count", 25),#每个节点的最大属性数
        )
    # 9. 执行异步扩展 (Core Execution)
    # 调用根节点的 expand_nodes_async 方法，开始递归地生成数据。
    # await 关键字表示这里会挂起当前函数，直到整个树的生成任务完成。
    # 结果会实时或分批写入 output_file 和 result_file。
    await root_node.expand_nodes_async(output_file=output_file, result_file=result_file)
    # 10. 保存树结构
    # 生成结束后，单独保存一份树的结构文件（通常只包含节点关系，不包含具体生成的样本数据），用于可视化或分析。
    root_node.save_tree_structure(tree_structure_file)
    # 11. 最终校验与返回
    # 检查根节点下是否有数据，如果生成失败（空数据），记录错误并返回 None。 todo generator_math_async生成,根节点并没有基于生成的 枢轴样本来确定划分准者和分类属性,而是人为规定
    # todo如果是 Root 节点，允许它没有 samples，只要它有 children 就行
    if not root_node.samples and not root_node.children:
        logger.error("Root node has no samples and no children after expansion.")
    elif not root_node.samples:
        logger.info("Root node expansion comp lete (Data acts as a container).")

    logger.info("Data generation completed successfully.")
    return root_node

if __name__ == "__main__":
    set_seed(42)

    #获取当前时间并生成时间戳字符串
    # datetime.datetime.now() 获取当前系统本地的日期和时间对象。
    # .strftime(...) 是 "String Format Time" 的缩写，用于将时间对象格式化为字符串。
    # 最终生成的字符串如 "20250106_123045"，保证了文件名的唯一性和时间顺序。
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # os.path.join(...) 根据操作系统自动处理路径分隔符。
    # 在 Windows 上会拼接成 "output\20250106_123045"，在 Linux/Mac 上是 "output/20250106_123045"
    output_dir = os.path.join("output", timestamp)
    os.makedirs(output_dir, exist_ok=True)#如果父目录 "output" 不存在，它会自动先创建父目录，再创建子目录。# exist_ok=True 是一个安全参数：如果该目录已经存在，程序不会报错（抛出 FileExistsError），而是静默跳过

    log_file = os.path.join(output_dir, "generation.log")#详细的生成日志
    output_file = os.path.join(output_dir, "output.json")#完整的树结构和中间结果
    tree_structure_file = os.path.join(output_dir, "tree_structure.txt")#人类可读的树结构可视化
    result_file = os.path.join(output_dir, "result.jsonl")#最终生成的数据样本（每行一个 JSON 对象）

    """通常用于断点续训或结果缓存的逻辑中。它检查是否指定了一个旧的结果文件并且该文件存在，如果满足条件，就直接复制过来跳过计算；否则，就开始新的计算任务"""
    # 1. 定义要加载的源文件路径
    # 如果设置为 None，表示不加载旧文件，强制重新开始。
    # 如果设置为具体路径字符串（例如 "output/20250101/data.json"），则尝试复用该文件。
    load_from = None
    # 2. 检查加载路径是否有效
    # load_from is not None: 首先确保变量被赋值了（不是 None）。
    # os.path.exists(load_from): 然后检查该路径在磁盘上是否真的存在文件。
    # 只有两个条件都为 True 时，才进入复用逻辑。
    if load_from is not None and os.path.exists(load_from):
        # 3. 打印日志提示
        print(f"[main] Found old output.json => copying from {load_from} to {output_file}")
        shutil.copyfile(load_from, output_file)
    else:
        # 5. 处理不需要复用或文件不存在的情况
        # 如果 load_from 是 None，或者指定的文件路径不存在，则执行此分支。
        print(f"[main] No old file found, will start fresh.")

    setup_logger(log_file)#配置并返回一个全局 logger 对象。功能：同时将日志输出到控制台（屏幕）和指定的文件中。
    
    # Run with vLLM backend
    #asyncio.run(
    #    run_generation_process(
    #        log_file=log_file,
    #        output_file=output_file,
    #        tree_structure_file=tree_structure_file,
    #        result_file=result_file,
    #        backend="vllm",
    #        use_api_pool=False
    #    )
    #)
    
    # Run with OpenAI backend
    """
    在 Python 中，异步函数（协程）有一个特性：你不能直接调用它。 如果你像普通函数那样写 run_generation_process(...)，它只会返回一个协程对象，而不会真正执行内部的代码。

    1. 核心含义：启动“事件循环”
    Python 的异步代码需要一个“调度员”来管理，这个调度员被称为 事件循环 (Event Loop)。
    
    asyncio.run() 的作用是：
    
    开启一个新的事件循环。
    
    运行传入的异步函数（主入口），直到它全部完成。
    
    关闭事件循环，清理所有残留任务。
    """
    asyncio.run(
        run_generation_process(
            log_file=log_file,
            output_file=output_file,
            tree_structure_file=tree_structure_file,
            result_file=result_file,
            backend="openai",
            use_api_pool=False
        )
    )