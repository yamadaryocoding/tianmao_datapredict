import pandas as pd
import os

def split_csv(file_path, chunk_size_mb=99, output_dir='split_output2'):
    """
    Splits a large CSV file into smaller chunks based on an approximate chunk size in MB.

    Args:
        file_path (str): The path to the large CSV file.
        chunk_size_mb (int): The desired approximate maximum size of each chunk in MB.
        output_dir (str): The directory where the split files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Estimate the number of rows per chunk
    # First, get the total file size
    try:
        total_file_size_bytes = os.path.getsize(file_path)
    except OSError as e:
        print(f"错误：无法获取文件大小 '{file_path}': {e}")
        return

    if total_file_size_bytes == 0:
        print(f"错误：文件 '{file_path}' 为空。")
        return

    # Get header to estimate row size
    try:
        header_df = pd.read_csv(file_path, nrows=1)
        if header_df.empty:
            print(f"错误：无法从 '{file_path}' 读取表头或文件为空。")
            return
        header_size_bytes = len(','.join(header_df.columns).encode('utf-8')) + len(os.linesep.encode('utf-8'))
        
        # Get size of a small sample of rows to estimate average row size
        sample_rows = 1000
        sample_df = pd.read_csv(file_path, nrows=sample_rows, skiprows=1, header=None)
        if sample_df.empty: # Fallback if less than sample_rows data rows
            sample_df = pd.read_csv(file_path, skiprows=1, header=None)
            if sample_df.empty:
                print(f"警告：文件 '{file_path}' 只有表头或没有数据行。将创建一个空数据文件或仅包含表头的文件。")
                 # Create a single file with just the header if no data rows exist
                output_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_part_1.csv")
                header_df.to_csv(output_file_path, index=False)
                print(f"已创建文件：'{output_file_path}' (仅包含表头)")
                return


        sample_data_size_bytes = sum(sample_df.memory_usage(index=True, deep=True)) - sample_df.memory_usage(index=True).sum() # More accurate for strings
        if sample_data_size_bytes == 0 and len(sample_df) > 0 : # if deep=True is not effective, try another way
             sample_data_size_bytes = sum(len(row.to_csv(header=False, index=False).encode('utf-8')) for _, row in sample_df.iterrows())


        if len(sample_df) > 0 :
            avg_row_size_bytes = sample_data_size_bytes / len(sample_df)
        else: # Only header
             avg_row_size_bytes = header_size_bytes #  A rough fallback

        if avg_row_size_bytes == 0: # Avoid division by zero if estimation is off
            print("警告：无法准确估算平均行大小，将使用默认的较大行数进行分块。")
            rows_per_chunk = 1000000 # A large default if row size estimation fails
        else:
            desired_chunk_size_bytes = chunk_size_mb * 1024 * 1024
            # Adjust desired chunk size for header in each file
            rows_per_chunk = int((desired_chunk_size_bytes - header_size_bytes) / avg_row_size_bytes)
        
        if rows_per_chunk <= 0:
            rows_per_chunk = 1 # Ensure at least one row per chunk if header is too large or estimation is off
            print(f"警告：估算的每块行数过小或为负 ({rows_per_chunk})。这可能是因为表头相对于目标块大小而言过大，或者行大小估算不准确。将每块设置为1行。")


    except pd.errors.EmptyDataError:
        print(f"错误：文件 '{file_path}' 为空或无法解析。")
        return
    except Exception as e:
        print(f"读取文件 '{file_path}' 时发生错误：{e}。将使用默认行数进行分块。")
        rows_per_chunk = 1000000 # Default chunk size in rows if estimation fails

    print(f"文件 '{os.path.basename(file_path)}' 的总大小: {total_file_size_bytes / (1024*1024):.2f} MB")
    print(f"目标分块大小: {chunk_size_mb} MB")
    print(f"估算每个分块的行数: {rows_per_chunk}")

    if rows_per_chunk == 0 :
        print("错误：计算出的每个分块的行数为0，无法分割。请检查文件或调整分块大小。")
        return

    chunk_count = 1
    try:
        for chunk_df in pd.read_csv(file_path, chunksize=rows_per_chunk):
            output_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_part_{chunk_count}.csv")
            chunk_df.to_csv(output_file_path, index=False)
            print(f"已创建文件: '{output_file_path}' (大小: {os.path.getsize(output_file_path) / (1024*1024):.2f} MB)")
            chunk_count += 1
        print(f"\n文件 '{file_path}' 已成功分割成 {chunk_count - 1} 个部分，存放在目录 '{output_dir}' 下。")
    except pd.errors.EmptyDataError:
        print(f"处理文件 '{file_path}' 时遇到空数据块，可能是文件末尾。")
    except Exception as e:
        print(f"分割文件 '{file_path}' 时发生错误: {e}")

# --- 使用示例 ---
# 假设您的 user_log_format1.csv 文件在当前工作目录下
# 如果不在，请提供完整路径

# 获取用户提供的文件名
# 默认是 user_log_format1.csv，但为了通用性，可以设置为变量
large_csv_file = 'train_all.csv' # 您可以更改为您的文件名

# 检查文件是否存在
if os.path.exists(large_csv_file):
    split_csv(large_csv_file)
else:
    print(f"错误: 文件 '{large_csv_file}' 未找到。请确保文件名正确且文件在指定路径中。")
    print(f"当前工作目录是: {os.getcwd()}")