import re


def process_text(text):
    # 使用正则表达式替换一个或多个空白字符为单个制表符
    return re.sub(r'([^\t\n\r\f\v])\s+([0-9])', r'\1\t\2', text)


def main():
    # 输入文件路径
    input_path = "E:\\WeChatFiles\\WeChat Files\\wxid_yugzcq6246xr22\\FileStorage\\File\\2024-05\\train.txt"
    # 输出文件路径
    output_path = "train_new.txt"

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 处理文件内容
    new_content = process_text(content)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"文件已处理，新的文件为: {output_path}")


if __name__ == "__main__":
    main()
