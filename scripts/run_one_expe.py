#################################################################################################################
#
# @author : Siyi Li (TU Braunschweig)
# @description : NILMFormer - Experiments
# 该脚本是整个项目的实验入口，用于：
#   1）解析命令行参数（数据集、电器、模型、窗口大小等）
#   2）加载 YAML 配置（configs/*.yaml）
#   3）构建并预处理 NILM 数据集
#   4）缓存预处理结果，避免重复耗时操作
#   5）调用训练与评估流程，输出实验结果
#
#################################################################################################################

# 标准库与第三方库导入：用于参数解析、文件路径操作、读取 YAML 配置以及日志记录
import argparse
import os
import yaml
import logging

# 关闭 torch flop counter 的日志，避免训练过程中产生多余无用输出
logging.getLogger("torch.utils.flop_counter").disabled = True

# 数值计算与深度学习框架
import numpy as np
import torch

# 配置管理库：将字典/YAML 转为易于访问的配置对象
from omegaconf import OmegaConf

# 项目内部工具函数与模块
from src.helpers.utils import create_dir
from src.helpers.preprocessing import (
    UKDALE_DataBuilder,
    REFIT_DataBuilder,
    split_train_test_nilmdataset,
    split_train_test_pdl_nilmdataset,
    nilmdataset_to_tser,
)
from src.helpers.dataset import NILMscaler
from src.helpers.expes import launch_models_training


def _configure_nilm_loss_hyperparams(expes_config, data, threshold):
    """
    根据当前电器的功率统计特性，自适应地设置 NILM 损失函数相关的超参数。

    主要包含：
      - on/off 占空比 duty_cycle
      - 噪声水平与边缘强度，用于设置梯度损失权重
      - 能量约束相关阈值（energy_floor_raw 等）
    这些量最终会写入 expes_config，供训练阶段读取。
    """
    try:
        # 要求输入为 4 维张量，且至少包含聚合 + 1 个电器，否则直接返回
        if data.ndim != 4 or data.shape[1] < 2:
            return
        # 取第一个电器通道的功率（索引 1，索引 0 是聚合）
        power = data[:, 1, 0, :].astype(np.float32)
        # 展平成一维，便于整体统计
        flat = power.reshape(-1)
    except Exception:
        return
    # 没有有效样本则无需配置
    if flat.size == 0:
        return
    # 将阈值转换为 float 以参与后续计算
    thr = float(threshold)
    # 大于阈值视为“开机”状态
    on_mask = flat > thr
    # 占空比：开机时间点比例
    duty_cycle = float(on_mask.mean())
    # 计算所有相邻时间点功率差，用于估计噪声与边缘强度
    if flat.size > 1:
        diff_all = np.abs(np.diff(flat))
    else:
        diff_all = np.zeros(1, dtype=np.float32)
    # 分别提取 on/off 区间
    on_values = flat[on_mask]
    off_values = flat[~on_mask]
    # on 区间内部的功率变化
    if on_values.size > 1:
        diff_on = np.abs(np.diff(on_values))
    else:
        diff_on = diff_all
    # off 区间内部的功率变化
    if off_values.size > 1:
        diff_off = np.abs(np.diff(off_values))
    else:
        diff_off = diff_all
    # 使用 90% 分位数估计噪声水平（优先使用 off 区间）
    if diff_off.size > 0:
        noise_level = float(np.quantile(diff_off, 0.9))
    else:
        noise_level = float(np.quantile(diff_all, 0.9))
    # 使用 90% 分位数估计边缘变化幅度（优先使用 on 区间）
    if diff_on.size > 0:
        edge_level = float(np.quantile(diff_on, 0.9))
    else:
        edge_level = float(np.quantile(diff_all, 0.9))
    # 边缘/噪声比，用于调节梯度损失权重
    ratio = edge_level / (noise_level + 1e-6)
    if not np.isfinite(ratio):
        ratio = 1.0
    # 裁剪到 [1, 10]，避免极端值导致权重过大或过小
    ratio_clipped = min(max(ratio, 1.0), 10.0)
    # 在 [0.2, 0.8] 之间插值得到 lambda_grad
    lambda_grad = 0.2 + (0.8 - 0.2) * (ratio_clipped - 1.0) / 9.0
    # duty_cycle 越小，说明电器越少开启，需要加大“开机样本”权重、减小“关机样本”权重
    if duty_cycle < 0.01:
        alpha_on = 4.0
        alpha_off = 0.5
    elif duty_cycle < 0.03:
        alpha_on = 3.5
        alpha_off = 0.7
    elif duty_cycle < 0.10:
        alpha_on = 3.0
        alpha_off = 1.0
    else:
        alpha_on = 2.0
        alpha_off = 1.0
    # 根据 duty_cycle 调整能量损失权重
    if duty_cycle < 0.01:
        lambda_energy = 0.02
    elif duty_cycle < 0.03:
        lambda_energy = 0.05
    elif duty_cycle < 0.10:
        lambda_energy = 0.10
    else:
        lambda_energy = 0.20

    # 计算 soft 阈值和平滑边缘项的原始尺度（后续会按缩放因子归一化）
    soft_temp_raw = max(0.25 * thr, 2.0 * noise_level, 1.0)
    edge_eps_raw = max(3.0 * noise_level, 0.5 * edge_level, 0.1 * thr, 1.0)

    try:
        # 计算每个时间窗的总能量
        energy_all = power.sum(axis=-1)
        if energy_all.size > 0:
            # 标记出“至少有一个时间点超过阈值”的窗口，视为开启窗口
            window_on = (power > thr).any(axis=-1)
            energy_on = energy_all[window_on]
            if energy_on.size > 0:
                # 开启窗口能量的 10% 分位数作为基准下界
                base_floor = float(np.quantile(energy_on, 0.1))
            else:
                # 若没有明显开启窗口，则退化为整体能量中位数
                base_floor = float(np.quantile(energy_all, 0.5))
            # 能量下限综合考虑 “阈值 * 窗长” 与 “典型开启能量”
            energy_floor_raw = max(
                0.1 * thr * power.shape[-1],
                0.05 * base_floor,
            )
        else:
            energy_floor_raw = thr * power.shape[-1] * 0.1
    except Exception:
        energy_floor_raw = thr * power.shape[-1] * 0.1

    # 将所有损失相关超参数写回配置对象，供训练阶段使用
    expes_config["loss_alpha_on"] = float(alpha_on)
    expes_config["loss_alpha_off"] = float(alpha_off)
    expes_config["loss_lambda_grad"] = float(lambda_grad)
    expes_config["loss_lambda_energy"] = float(lambda_energy)
    expes_config["loss_soft_temp_raw"] = float(soft_temp_raw)
    expes_config["loss_edge_eps_raw"] = float(edge_eps_raw)
    expes_config["loss_energy_floor_raw"] = float(energy_floor_raw)


def get_cache_path(expes_config: OmegaConf):
    """
    根据实验配置构造一个唯一 key，用于生成缓存文件路径。

    相同的：
      - 数据集
      - 电器
      - 采样率
      - 窗口大小
      - 随机种子
      - 归一化方式
    会映射到同一个缓存文件，避免重复做数据预处理。
    """
    if getattr(expes_config, "name_model", None) == "DiffNILM":
        # 对 DiffNILM 模型，额外在 key 中加入模型名称，防止与其他模型混淆
        key_elements = [
            expes_config.dataset,
            expes_config.appliance,
            expes_config.sampling_rate,
            str(expes_config.window_size),
            str(expes_config.seed),
            expes_config.power_scaling_type,
            expes_config.appliance_scaling_type,
            "DiffNILM",
        ]
    else:
        # 其他模型只使用通用的几个字段构造 key
        key_elements = [
            expes_config.dataset,
            expes_config.appliance,
            expes_config.sampling_rate,
            str(expes_config.window_size),
            str(expes_config.seed),
            expes_config.power_scaling_type,
            expes_config.appliance_scaling_type,
        ]
    # 将各个元素用 "_" 拼接成字符串 key
    key = "_".join(str(x) for x in key_elements)
    # 为防止 key 中带有路径分隔符，先做一次替换
    key = key.replace("/", "-")
    # 所有缓存文件都放在 data_cache 目录下
    cache_dir = os.path.join("data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    # 返回完整的缓存文件路径
    return os.path.join(cache_dir, key + ".pt")


def launch_one_experiment(expes_config: OmegaConf):
    """
    在给定配置下执行一次完整的实验流程：
      1）尝试从缓存中加载预处理好的数据；
      2）若无缓存，则构建原始数据集并做预处理/划分；
      3）根据归一化结果设置损失函数相关参数；
      4）将预处理结果写入缓存；
      5）调用统一的训练入口 launch_models_training。
    """
    # 固定 numpy 随机种子，保证数据划分等过程的可复现性
    np.random.seed(seed=expes_config.seed)

    # 生成当前配置对应的缓存路径
    cache_path = get_cache_path(expes_config)
    # 如果缓存文件存在，则直接加载，跳过耗时的数据预处理
    if os.path.isfile(cache_path):
        logging.info("Load cached preprocessed data from %s", cache_path)
        # 加载缓存：包含预处理后的数据 tuple、缩放器、cutoff 与 threshold
        cache = torch.load(cache_path, weights_only=False)
        tuple_data = cache["tuple_data"]
        scaler = cache["scaler"]
        expes_config.cutoff = cache["cutoff"]
        expes_config.threshold = cache["threshold"]
        # 使用缓存数据直接启动模型训练
        return launch_models_training(tuple_data, scaler, expes_config)

    # 若不存在缓存，则从头开始构建数据
    logging.info("Process data ...")
    if expes_config.dataset == "UKDALE":
        # 针对 UKDALE 数据集构造数据构建器
        data_builder = UKDALE_DataBuilder(
            data_path=f"{expes_config.data_path}/UKDALE/",
            mask_app=expes_config.app,
            sampling_rate=expes_config.sampling_rate,
            window_size=expes_config.window_size,
        )

        # 先在固定房屋集合 [1,2,3,4,5] 上构建整体 NILM 数据，用于统计全局阈值等
        data, st_date = data_builder.get_nilm_dataset(house_indicies=[1, 2, 3, 4, 5])

        # 若 window_size 为字符串（如 "day"），则以 builder 内部计算出的实际窗口长度为准
        if isinstance(expes_config.window_size, str):
            expes_config.window_size = data_builder.window_size

        # 使用配置中指定的训练房屋索引构造训练数据
        data_train, st_date_train = data_builder.get_nilm_dataset(
            house_indicies=expes_config.ind_house_train
        )
        # 使用配置中指定的测试房屋索引构造测试数据
        data_test, st_date_test = data_builder.get_nilm_dataset(
            house_indicies=expes_config.ind_house_test
        )

        # 在训练房屋集合内部再划分出验证集（perc_house_test 比例的房屋作为 valid）
        data_train, st_date_train, data_valid, st_date_valid = (
            split_train_test_nilmdataset(
                data_train,
                st_date_train,
                perc_house_test=0.2,
                seed=expes_config.seed,
            )
        )

    elif expes_config.dataset == "REFIT":
        # 针对 REFIT 数据集构造数据构建器
        data_builder = REFIT_DataBuilder(
            data_path=f"{expes_config.data_path}/REFIT/RAW_DATA_CLEAN/",
            mask_app=expes_config.app,
            sampling_rate=expes_config.sampling_rate,
            window_size=expes_config.window_size,
        )

        # 基于 house_with_app_i 中的房屋索引构建整体 NILM 数据
        data, st_date = data_builder.get_nilm_dataset(
            house_indicies=expes_config.house_with_app_i
        )

        # 同样处理字符串形式的 window_size
        if isinstance(expes_config.window_size, str):
            expes_config.window_size = data_builder.window_size

        # 按 PDL 思路：先从所有房屋中划分出测试集房屋
        data_train, st_date_train, data_test, st_date_test = (
            split_train_test_pdl_nilmdataset(
                data.copy(), st_date.copy(), nb_house_test=2, seed=expes_config.seed
            )
        )

        # 再在剩余房屋中划分出验证集房屋
        data_train, st_date_train, data_valid, st_date_valid = (
            split_train_test_pdl_nilmdataset(
                data_train, st_date_train, nb_house_test=1, seed=expes_config.seed
            )
        )

    logging.info("             ... Done.")

    # 从 data_builder 中读取当前电器的最小功率阈值（用于 on/off 判定）
    threshold = data_builder.appliance_param[expes_config.app]["min_threshold"]
    expes_config.threshold = threshold
    # 基于原始功率序列和阈值，自动配置与 NILM 损失相关的若干超参数
    _configure_nilm_loss_hyperparams(expes_config, data, threshold)

    # 构造缩放器，用于对聚合功率及电器功率做归一化（标准化/最大值缩放等）
    scaler = NILMscaler(
        power_scaling_type=expes_config.power_scaling_type,
        appliance_scaling_type=expes_config.appliance_scaling_type,
    )
    # 在整体数据上拟合缩放器并完成缩放
    data = scaler.fit_transform(data)

    # cutoff 记录电器功率尺度（如最大值），后续将阈值等换算到无量纲空间
    expes_config.cutoff = float(scaler.appliance_stat2[0])
    if expes_config.cutoff and expes_config.cutoff > 0:
        # 将功率阈值按 cutoff 缩放，得到 loss_threshold
        expes_config["loss_threshold"] = float(expes_config.threshold) / float(
            expes_config.cutoff
        )
        if "loss_soft_temp_raw" in expes_config:
            # soft 温度同样做缩放
            expes_config["loss_soft_temp"] = float(expes_config.loss_soft_temp_raw) / float(
                expes_config.cutoff
            )
        if "loss_edge_eps_raw" in expes_config:
            # 边缘平滑参数做缩放
            expes_config["loss_edge_eps"] = float(expes_config.loss_edge_eps_raw) / float(
                expes_config.cutoff
            )

    # 对 ConvNet/ResNet/Inception 等时间序列分类/回归基线，先将 NILM 4D 数据转换为 2D TS 任务格式
    if expes_config.name_model in ["ConvNet", "ResNet", "Inception"]:
        # 在整体数据上转换出 (X, y) 形式，便于后续评估/可视化
        X, y = nilmdataset_to_tser(data)

        # 分别对 train/valid/test 做缩放
        data_train = scaler.transform(data_train)
        data_valid = scaler.transform(data_valid)
        data_test = scaler.transform(data_test)

        # 再将缩放后的 NILM 4D 数据转换为 TSER 格式
        X_train, y_train = nilmdataset_to_tser(data_train)
        X_valid, y_valid = nilmdataset_to_tser(data_valid)
        X_test, y_test = nilmdataset_to_tser(data_test)

        # 打包四个分割（train/valid/test/整体），供下游训练与评估使用
        tuple_data = (
            (X_train, y_train, st_date_train),
            (X_valid, y_valid, st_date_valid),
            (X_test, y_test, st_date_test),
            (X, y, st_date),
        )

    else:
        # 对 NILM 模型（如 NILMFormer、UNetNILM 等），直接在 4D NILM 张量上进行缩放
        data_train = scaler.transform(data_train)
        data_valid = scaler.transform(data_valid)
        data_test = scaler.transform(data_test)

        # 将划分后的数据及对应起始时间封装到一个统一的 tuple 中
        tuple_data = (
            data_train,
            data_valid,
            data_test,
            data,
            st_date_train,
            st_date_valid,
            st_date_test,
            st_date,
        )

    # 将预处理好的数据、缩放器及阈值信息写入缓存文件，供下次相同配置直接复用
    cache = {
        "tuple_data": tuple_data,
        "scaler": scaler,
        "cutoff": expes_config.cutoff,
        "threshold": expes_config.threshold,
    }
    torch.save(cache, cache_path)

    # 最后调用统一的训练入口，根据模型名称与配置训练并评估模型
    return launch_models_training(tuple_data, scaler, expes_config)


def main(
    dataset,
    sampling_rate,
    window_size,
    appliance,
    name_model,
    resume,
    no_final_eval,
    loss_type=None,
):
    """
    Main function to load configuration, update it with parameters,
    and launch an experiment.

    Args:
        dataset (str): Name of the dataset (case-insensitive, e.g. UKDALE or REFIT).
        sampling_rate (str): Selected sampling rate (case-insensitive, e.g. 30s, 1min).
        window_size (int or str): Size of the window (converted to int if possible not day, week or month).
        appliance (str): Selected appliance (case-insensitive).
        name_model (str): Name of the model to use for the experiment (case-insensitive).
    """

    # 固定随机种子（目前只用于日志记录和某些下游过程，主要数据划分由 DataBuilder 内部控制）
    seed = 42

    try:
        # window_size 既可能是数字（如 "256"），也可能是字符串（如 "day"/"week"）
        # 这里优先尝试将其解析为整数；解析失败则保留原始字符串并记录 warning
        window_size = int(window_size)
    except ValueError:
        logging.warning(
            "window_size could not be converted to int. Using its original value: %s",
            window_size,
        )

    # 读取基础实验配置，包含：
    #   - 数据路径 data_path
    #   - 结果路径 result_path
    #   - 批大小/训练轮数等通用训练参数
    with open("configs/expes.yaml", "r") as f:
        expes_config = yaml.safe_load(f)

    # 读取数据集配置，用于检查数据集名称并获取电器相关配置
    with open("configs/datasets.yaml", "r") as f:
        datasets_all = yaml.safe_load(f)
        dataset_key_map = {k.lower(): k for k in datasets_all.keys()}
        # 使用忽略大小写的映射，从命令行传入的 dataset 字符串得到规范化 key
        dataset_key = dataset_key_map.get(str(dataset).strip().lower())
        if dataset_key is None:
            # 收集所有可用数据集名称，用于错误提示
            available = ", ".join(sorted(datasets_all.keys()))
            raise ValueError(
                "Dataset {} unknown. Available datasets (case-insensitive): {}. Use -h to see argument help.".format(
                    dataset, available
                )
            )
        # dataset_key 对应的数据集部分配置（例如 UKDALE 下有哪些电器）
        datasets_config = datasets_all[dataset_key]

    # 读取所有模型的配置，用于根据 name_model 选择对应超参数
    with open("configs/models.yaml", "r") as f:
        baselines_config = yaml.safe_load(f)

        model_key_map = {k.lower(): k for k in baselines_config.keys()}
        # 将命令行传入的模型名映射为配置中的规范模型 key
        model_key = model_key_map.get(str(name_model).strip().lower())
        if model_key is None:
            # 若模型名非法，同样给出所有可用模型名称
            available = ", ".join(sorted(baselines_config.keys()))
            raise ValueError(
                "Model {} unknown. Available models (case-insensitive): {}. Use -h to see argument help.".format(
                    name_model, available
                )
            )
        # 将模型对应的配置（model_kwargs + 训练参数等）合并到 expes_config 中
        expes_config.update(baselines_config[model_key])

    # 针对当前数据集，构造电器名称的忽略大小写映射
    appliance_key_map = {k.lower(): k for k in datasets_config.keys()}
    appliance_key = appliance_key_map.get(str(appliance).strip().lower())
    if appliance_key is None:
        # 电器名称非法，记录错误日志并抛出异常，同时列出所有可选电器名
        available = ", ".join(sorted(datasets_config.keys()))
        logging.error("Appliance '%s' not found in datasets_config.", appliance)
        raise ValueError(
            "Appliance {} unknown for dataset {}. Available appliances (case-insensitive): {}. Use -h to see argument help.".format(
                appliance, dataset_key, available
            )
        )
    # 合并当前电器的配置（如 house_with_app_i / ind_house_train 等）到实验配置
    expes_config.update(datasets_config[appliance_key])

    # 将采样率规范化为小写字符串，便于后续比较
    sampling_rate = str(sampling_rate).strip().lower()

    # 打印一段清晰的实验摘要到日志，方便之后快速回顾本次实验设置
    logging.info("---- Run experiments with provided parameters ----")
    logging.info("      Dataset: %s", dataset_key)
    logging.info("      Sampling Rate: %s", sampling_rate)
    logging.info("      Window Size: %s", window_size)
    logging.info("      Appliance : %s", appliance_key)
    logging.info("      Model: %s", model_key)
    logging.info("      Seed: %s", seed)
    logging.info("--------------------------------------------------")

    # 将规范化后的字段写回 expes_config，作为后续统一的数据源
    expes_config["dataset"] = dataset_key
    expes_config["appliance"] = appliance_key
    expes_config["window_size"] = window_size
    expes_config["sampling_rate"] = sampling_rate
    expes_config["seed"] = seed
    expes_config["name_model"] = model_key
    expes_config["resume"] = bool(resume)
    expes_config["skip_final_eval"] = bool(no_final_eval)
    # 如在命令行显式指定 loss_type，则写入配置（覆盖默认设置）
    if loss_type is not None:
        expes_config["loss_type"] = str(loss_type)

    # 逐级创建结果目录：
    #   result_path / {dataset}_{sampling_rate} / {window_size} /
    result_path = create_dir(expes_config["result_path"])
    result_path = create_dir(f"{result_path}{dataset_key}_{sampling_rate}/")
    result_path = create_dir(f"{result_path}{window_size}/")

    expes_config = OmegaConf.create(expes_config)

    # 保存最终结果（模型、指标等）时使用的基础前缀路径：
    #   .../{model_name}_{seed}
    expes_config.result_path = (
        f"{result_path}{expes_config.name_model}_{expes_config.seed}"
    )

    if torch.cuda.is_available():
        # 如当前环境有 CUDA，则尝试提升 float32 矩阵乘法的数值精度
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # 最终调用 launch_one_experiment，根据当前配置执行一次完整实验
    launch_one_experiment(expes_config)


if __name__ == "__main__":
    # 仅当脚本被直接运行（而非以模块导入）时，才解析命令行参数
    # 首先读取所有数据集和模型配置，用于构造帮助信息
    with open("configs/datasets.yaml", "r") as f:
        _datasets_all = yaml.safe_load(f)
    with open("configs/models.yaml", "r") as f:
        _models_all = yaml.safe_load(f)
    # 将所有可用数据集与模型名拼接成字符串，用于 argparse help 文本
    _dataset_choices = ", ".join(sorted(_datasets_all.keys()))
    _model_choices = ", ".join(sorted(_models_all.keys()))
    _appliance_hints = []
    if "REFIT" in _datasets_all:
        _appliance_hints.append(
            "REFIT: " + ", ".join(sorted(_datasets_all["REFIT"].keys()))
        )
    if "UKDALE" in _datasets_all:
        _appliance_hints.append(
            "UKDALE: " + ", ".join(sorted(_datasets_all["UKDALE"].keys()))
        )
    # 将不同数据集的电器列表拼接成一行提示字符串，用于 CLI 帮助信息
    _appliance_help = " | ".join(_appliance_hints) if _appliance_hints else ""

    # 构造命令行解析器，定义所有可供用户设置的实验参数
    parser = argparse.ArgumentParser(
        description=(
            "NILMFormer Experiments. Use -h to see valid options for each argument."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Dataset name (non-case-insensitive). Choices: {}.".format(_dataset_choices),
    )
    parser.add_argument(
        "--sampling_rate",
        required=True,
        type=str,
        help="Sampling rate (non-case-insensitive), e.g. '30s', '1min', '10min'.",
    )
    parser.add_argument(
        "--window_size",
        required=True,
        type=str,
        help="Window size used for training, e.g. '128' or 'day.",
    )
    parser.add_argument(
        "--appliance",
        required=True,
        type=str,
        help=(
            "Selected appliance (non-case-insensitive). Available by dataset: {}.".format(
                _appliance_help
            )
        ),
    )
    parser.add_argument(
        "--name_model",
        required=True,
        type=str,
        help="Name of the model for training (non-case-insensitive). Choices: {}.".format(
            _model_choices
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing checkpoint for the same experiment if available.",
    )
    parser.add_argument(
        "--no_final_eval",
        action="store_true",
        help="Skip final full evaluation (keep visualization HTML only).",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        help=(
            "Loss type for NILM baselines. Choices: "
            "'nilm_composite', 'eaec', 'smoothl1', 'mse', 'mae'."
        ),
    )

    # 解析命令行参数，并将其传入 main 函数启动一次实验
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        appliance=args.appliance,
        name_model=args.name_model,
        resume=args.resume,
        no_final_eval=args.no_final_eval,
        loss_type=args.loss_type,
    )
