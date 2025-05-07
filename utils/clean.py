import pandas as pd

from utils.config import PRE_DATA_PATH, REQ_CSV_CSV_PATH
from utils.logger import logger


def clean_csv(input_file, output_file, encoding="UTF-8-SIG"):
    # 打开原始文件读取内容
    with open(input_file, encoding=encoding) as infile:
        lines = infile.readlines()

    # 打开目标文件写入内容
    with open(output_file, "w", encoding=encoding) as outfile:
        # 记录是否已经写入过标题行
        header_written = False

        for line in lines:
            if line.endswith(",0\n"):
                continue
            if "时间点" in line:
                # 如果是首次遇到标题行，写入它
                if not header_written:
                    outfile.write(line)
                    header_written = True
            else:
                # 写入非标题行
                outfile.write(line)
            # 删除最后一列为 0 的行
    # 处理完成后读取文件，将时间点从 0 开始
    df = pd.read_csv(output_file)
    df["时间点"] = df["时间点"] - df["时间点"].min().item()
    df.to_csv(output_file, index=False)

    logger.info(f"File cleaned and saved as {output_file}")


for month_dir in REQ_CSV_CSV_PATH.iterdir():
    month = month_dir.name
    logger.info(f"Processing Month：{month}")
    month_cleaned_path = PRE_DATA_PATH.joinpath("csv").joinpath(f"{month}")
    month_cleaned_path.mkdir(parents=True, exist_ok=True)

    for daily_file in month_dir.iterdir():
        date = daily_file.name.split(".")[0]
        input_file = daily_file
        output_file = month_cleaned_path.joinpath(f"{date}.csv")
        clean_csv(input_file, output_file)
