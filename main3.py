import os
import shutil
from xml.etree import ElementTree as ET
import math
import re

def rotate_svg(svg_path, output_path):
    # 回転関数
    def rotate_point_around_center(x, y, cx, cy, angle_rad):
        """点(x, y)を中心(cx, cy)周りにangle_rad回転する"""
        x -= cx
        y -= cy
        new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad) + cx
        new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad) + cy
        return new_x, new_y

    # パスデータを回転
    def rotate_path_data(path_data, cx, cy, angle_rad):
        """パスデータ(d属性)内の座標を回転する"""
        def replace_coords(match):
            x, y = map(float, match.groups())
            new_x, new_y = rotate_point_around_center(x, y, cx, cy, angle_rad)
            return f"{new_x:.3f},{new_y:.3f}"

        # 座標の正規表現に一致する部分を変換
        path_re = re.compile(r"([+-]?\d*\.?\d+),([+-]?\d*\.?\d+)")
        return path_re.sub(replace_coords, path_data)


    # SVGファイルを読み込む
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # 名前空間を削除する
    for elem in tree.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]  # 名前空間を削除

    # キャンバスサイズを取得
    width_attr = root.attrib.get("width", "0")
    height_attr = root.attrib.get("height", "0")
    
    # Remove 'px' or other units if present
    width = float(width_attr.replace('px', '').replace('em', '').replace('cm', '').replace('mm', '').strip())
    height = float(height_attr.replace('px', '').replace('em', '').replace('cm', '').replace('mm', '').strip())
    
    viewBox = root.attrib.get("viewBox", None)
    if viewBox:
        _, _, width, height = map(float, viewBox.split())
    
    cx, cy = width / 2, height / 2  # キャンバスの中心
    angle_rad = math.pi / 2  # 反時計回り90度（ラジアン）    

    # パス要素を回転
    for element in root.iter('path'):  # 名前空間なしで直接pathタグを指定
        d = element.attrib.get('d', '')
        if d:
            element.set('d', rotate_path_data(d, cx, cy, angle_rad))

    # 保存
    tree.write(output_path)


def char_to_unicode_map(chars):
    """
    指定された文字リストをUnicode名に変換する辞書を作成する。
    """
    return {char: f"uni{ord(char):04X}" for char in chars}

def process_svgs(input_dir, output_dir, r_vert_path, vert_path):
    os.makedirs(output_dir, exist_ok=True)

    # r_vert.txtとvert.txtを読み込む
    with open(r_vert_path, 'r', encoding='utf-8') as f:
        r_vert_chars = set(f.read().strip())

    with open(vert_path, 'r', encoding='utf-8') as f:
        vert_chars = set(f.read().strip())

    # r_vert_charsとvert_charsの重複を確認
    overlap = r_vert_chars & vert_chars
    if overlap:
        raise ValueError(f"r_vert.txtとvert.txtの両方に含まれる文字があります: {''.join(overlap)}")

    # 文字からUnicode名へのマッピングを作成
    r_vert_map = char_to_unicode_map(r_vert_chars)
    vert_map = char_to_unicode_map(vert_chars)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.svg'):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 元のSVGを出力ディレクトリにコピー
        shutil.copyfile(input_path, output_path)

        # ファイル名からUnicode名を取得
        base_name = os.path.splitext(filename)[0]

        # r_vert処理対象の場合
        if base_name in r_vert_map.values():
            rotated_output_path = os.path.join(output_dir, f"{base_name}.vert.svg")
            rotate_svg(input_path, rotated_output_path)

        # vert処理対象の場合
        elif base_name in vert_map.values():
            vert_output_path = os.path.join(output_dir, f"{base_name}.vert.svg")
            shutil.copyfile(input_path, vert_output_path)

if __name__ == "__main__":
    input_directory = "06_output_images"
    output_directory = "07_output_images"
    r_vert_file = "r_vert.txt"
    vert_file = "vert.txt"

    process_svgs(input_directory, output_directory, r_vert_file, vert_file)
