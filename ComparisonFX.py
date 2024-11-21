# ファイルパスを設定
file_a = "font_settings.txt"
file_b = "Comparison.txt"

# ファイルを読み込む関数
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# AとBの文字を比較してAにない文字を検出する関数
def find_extra_characters(text_a, text_b):
    set_a = set(text_a)  # Aの文字集合
    set_b = set(text_b)  # Bの文字集合
    extra_characters = set_b - set_a  # BにあってAにない文字
    return extra_characters

# ファイルを読み込む
text_a = read_file(file_a)
text_b = read_file(file_b)

# Aにない文字を検出
extra_characters = find_extra_characters(text_a, text_b)

# 結果を表示
if extra_characters:
    print("不足文字:")
    print("".join(extra_characters))
else:
    print("不足なし")
