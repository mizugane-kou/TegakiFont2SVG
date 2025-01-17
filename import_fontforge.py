import fontforge
import os
import re

svg_path = "/your_SVG_path"

# 現在開いているフォントを取得
font = fontforge.activeFont()
if not font:
    raise RuntimeError("No font is currently open in FontForge.")

# フォントのエンコーディングをUnicodeに設定
font.encoding = "UnicodeFull"

# SVGファイル名の形式に一致する正規表現 (例: uni0041.svg)
svg_filename_pattern = re.compile(r"uni([0-9A-Fa-f]{4,6})(\.vert)?\.svg")

# 縦書きフォントに設定する幅
VERT_WIDTH = 1000  # 任意の幅に設定

# フォルダ内のSVGファイルをスキャン
for filename in os.listdir(svg_path):
    match = svg_filename_pattern.match(filename)
    if match:
        # 縦書きの場合、codepointを-1に設定
        if match.group(2) == ".vert":
            codepoint = -1
            glyph_name = "uni" + match.group(1) + ".vert"  # uni + Unicode値 + .vert
        else:
            codepoint = int(match.group(1), 16)  # 16進数から整数に変換
            glyph_name = "uni" + match.group(1)  # uni + Unicode値
        
        svg_filepath = os.path.join(svg_path, filename)

        print(f"Importing {filename} for Unicode codepoint {('U+' + match.group(1)) if codepoint != -1 else 'Vertical'}")

        # Unicodeコードポイントでグリフを作成
        glyph = font.createChar(codepoint, glyph_name)  # 名前も指定してグリフを作成
        glyph.importOutlines(svg_filepath)  # SVGをインポート
        
        # 縦書きの場合、幅を指定
        if codepoint == -1:  # 縦書き用グリフ
            glyph.width = VERT_WIDTH
            print(f"Set width of vertical glyph {glyph_name} to {VERT_WIDTH}")
    else:
        print(f"Skipping file {filename}: not a valid Unicode SVG")
