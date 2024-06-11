import os

# 指定要遍历的目录
directory = './annotations'

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    # 检查文件扩展名是否是.jpg
    if filename.endswith('.txt'):
        # 分割原始文件名，找到需要重命名的部分
        parts = filename.split('_')
        if len(parts) > 1:
            # 重命名规则，将日期和时间戳部分提取出来作为新文件名
            # new_filename = parts[-1].replace('.', '') + '.jpg'
            new_filename = (parts[1]+parts[2]).replace('.','',1)
            # 构造完整的原始文件路径和新文件路径
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # 重命名文件
            os.rename(old_file, new_file)
            print(f'Renamed "{filename}" to "{new_filename}"')

print('批量重命名完成。')
