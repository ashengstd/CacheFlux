# 环境配置

## Pixi

安装[Pixi](https://pixi.sh/latest//)进行项目环境管理

```shell
# MacOs and Linux:
curl -fsSL https://pixi.sh/install.sh | sh

# Windows:
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

## 使用 pixi install 搭建环境

```shell
pixi install  # 根据需要选择环境 cpu: pixi install -e cpu
pixi shell # 激活环境
```

# 运行

## 项目配置

参考`config.toml`配置路径

## 数据预处理

```shell
python utils/clean.py
```

```shell
python utils/npy_data.py
```

```shell
python droo_data.py
```

## 训练

```shell
python train.py
```

## 推理

```shell
python infer.py
```
