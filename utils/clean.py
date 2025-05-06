import pandas as pd
from constants import DATA_PATH, MONTH_SUFFIX_CLEANED, PRE_DATA_PATH


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

    print(f"文件处理完成，已保存为 {output_file}")


for month_dir in DATA_PATH.iterdir():
    if month_dir.name.endswith("月用户带宽数据"):
        month = month_dir.name.split("月")[0]
        month_cleaned_path = PRE_DATA_PATH.joinpath(f"{month}{MONTH_SUFFIX_CLEANED}")
        month_cleaned_path.mkdir(parents=True, exist_ok=True)

        for daily_file in month_dir.iterdir():
            date = daily_file.name[5:15]
            input_file = daily_file
            output_file = month_cleaned_path.joinpath(f"{date}.csv")
            clean_csv(input_file, output_file)
