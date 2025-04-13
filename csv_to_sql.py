import sqlite3

import pandas as pd
from rich.console import Console  # 导入 rich 的控制台输出模块
from rich.progress import Progress  # 导入 rich 的进度条模块
from utils.constants import DATA_PATH, PRE_DATA_PATH

# 创建 Console 实例
console = Console()


def clean_and_store_to_sqlite(input_file, db_connection):
    # 读取 CSV 文件
    df = pd.read_csv(input_file, header=0)

    # 删除其他重复的标题行（如果有）
    df = df[~df.apply(lambda x: x.str.contains("时间点").any(), axis=1)]
    df["时间点"] = df["时间点"].astype(int)

    # 删除所有值为 0 的行（假设删除整个行中任何为 0 的值）
    df = df[(df != 0).all(axis=1)]

    # 处理时间点，删除最小时间点后，使时间点从 0 开始
    df["时间点"] = df["时间点"] - df["时间点"].min()

    # 将数据写入 SQLite 数据库，假设你有一个名为 `user_bandwidth_data` 的表
    df.to_sql("user_bandwidth_data", db_connection, if_exists="replace", index=False)

    # 使用 rich 的 console 打印处理完成的消息
    console.print(
        f"[green]文件 {input_file} 处理完成，并已保存到 SQLite 数据库[/green]"
    )


def process_directory_to_sqlite():
    # 创建 rich 的进度条实例
    with Progress(console=console) as progress:
        # 遍历数据目录
        for month_dir in DATA_PATH.iterdir():
            if month_dir.name.endswith("月用户带宽数据"):
                month = month_dir.name.split("月")[0]
                # 创建输出目录
                output_month_dir = PRE_DATA_PATH.joinpath(month)
                output_month_dir.mkdir(parents=True, exist_ok=True)

                # 获取文件列表
                daily_files = list(month_dir.iterdir())

                # 为每个月的文件创建一个进度条
                task = progress.add_task(
                    f"Processing {month} month files", total=len(daily_files)
                )

                # 遍历每个文件
                for daily_file in daily_files:
                    date = daily_file.name[5:15]
                    input_file = daily_file

                    # 创建 SQLite 数据库连接
                    db_path = output_month_dir.joinpath(f"{date}.db")
                    db_connection = sqlite3.connect(db_path)

                    # 清洗数据并存储到 SQLite
                    clean_and_store_to_sqlite(input_file, db_connection)

                    # 关闭 SQLite 数据库连接
                    db_connection.close()

                    # 更新进度条
                    progress.update(task, advance=1)

        # 使用 rich 的 console 打印处理完成的消息
        console.print(
            "[bold green]所有数据处理完成，并已保存到 SQLite 数据库[/bold green]"
        )


if __name__ == "__main__":
    process_directory_to_sqlite()
