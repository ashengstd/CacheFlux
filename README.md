# droo 与线性规划的连接

# 线性规划 or 非线性规划

# UI or APP

# 环境配置

## Astral-uv

安装[Astral-uv](https://docs.astral.sh/uv/getting-started/installation/)进行项目环境管理

```shell
# MacOs and Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 使用 uv sync 搭建环境

```shell
uv sync --extra cu124/cpu # 根据显卡类型选择
```

# 运行

## Windows

使用 `Powershell` / `Microsoft Powershell`

```pwsh
./train.ps1
```

如果遇到脚本不可执行，可能是因为权限问题，运行下面这个

```pwsh
Set-ExecutionPolicy RemoteSigned
```

## Linux

POSIX shell or bash or Fish:

```shell
${shell} train.sh
```

# 项目常量定义

本项目使用 `utils/constants.py` 文件来定义一些常量路径和参数，以便在代码中统一管理和引用。这些常量主要用于指定数据文件的存储路径和其他相关配置。

## 常量定义

关于 `constants.py` 文件中定义的常量及其用途：

- `USERS`,`CACHES`,其中`CACHES`和模型的最后一层对应
- `N`：网络输出的方案的数量
- `DATA_PATH`：存储质量约束下的成本调度数据的路径。
- `PRE_DATA_PATH`：存储预处理数据的路径，主要是节点和 Cache 组的信息。
- `INPUT_DATA_PATH`：存储 DROO 输入数据的路径。
- `GLOBAL_PATH`：存储单纯形法相关数据的路径。
- `BEST_SOLUTION_PATH`：存储最优解的路径。
