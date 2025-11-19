import os
import argparse

def delete_files_from_txts(txt_path, target_dir, recursive=True):
    """
    根据txt文件中的文件名，递归删除目标目录及其子目录中的对应文件
    
    参数:
        txt_path: txt文件或包含txt文件的目录路径
        target_dir: 要删除文件的目标目录
        recursive: 是否递归查找txt文件
    """
    # 验证目标目录是否存在
    if not os.path.isdir(target_dir):
        print(f"错误：目标目录不存在 - {target_dir}")
        return

    # 收集所有要处理的txt文件
    txt_files = []
    if os.path.isfile(txt_path) and txt_path.lower().endswith('.txt'):
        txt_files.append(txt_path)
    elif os.path.isdir(txt_path):
        for root, dirs, files in os.walk(txt_path):
            for file in files:
                if file.lower().endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
            if not recursive:
                break  # 不递归则只处理顶层目录
    else:
        print("错误：输入的txt路径既不是文件也不是目录")
        return

    # 处理每个txt文件
    total_deleted = 0
    for txt_file in txt_files:
        print(f"\n处理文件: {txt_file}")
        deleted_count = 0
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            filename = line.strip()
            if not filename:  # 跳过空行
                continue
                
            # 在目标目录及其子目录中查找文件
            file_found = False
            for root, dirs, files in os.walk(target_dir):
                file_to_delete = os.path.join(root, filename)
                if os.path.exists(file_to_delete):
                    try:
                        os.remove(file_to_delete)
                        deleted_count += 1
                        file_found = True
                        print(f"已删除: {file_to_delete}")
                        break  # 找到并删除后跳出当前循环
                    except Exception as e:
                        print(f"删除失败 {file_to_delete}: {e}")
                        file_found = True
                        break
            
            if not file_found:
                print(f"文件不存在: {filename}")
        
        print(f"从此txt中删除了 {deleted_count} 个文件")
        total_deleted += deleted_count
    
    print(f"\n总共删除了 {total_deleted} 个文件")

if __name__ == "__main__":
    txt_path = r"/home/xuk/xuke/demos/SSD_2-main/test_FSD/untrust/high_ber_files.txt"  # 输入txt文件路径
    target_dir = r"/home/xuk/xuke/dataset/FSD/test/"  # 目标目录路径

    delete_files_from_txts(txt_path, target_dir, True)