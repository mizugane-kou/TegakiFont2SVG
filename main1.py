import cv2
import numpy as np
import os
from cv2 import aruco
import qrcode
from PIL import Image, ImageFont, ImageDraw, ImageOps


def setup_image_dimensions():
    # A4サイズの画像 (210mm x 297mm) を350dpiでピクセルに変換
    width = int(210 / 25.4 * 350)  # 幅は約2896
    height = int(297 / 25.4 * 350)  # 高さは約4101
    return width, height

def create_blank_image(width, height):
    # 白いA4サイズの画像を作成
    return np.ones((height, width, 3), dtype=np.uint8) * 255

def generate_qr_code(data, scale_factor, width, height):
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    qr_img = cv2.cvtColor(np.array(qr_img), cv2.COLOR_RGB2BGR)
    qr_scale = scale_factor / qr_img.shape[1]  # 画像の10%の大きさ
    return cv2.resize(qr_img, None, fx=qr_scale, fy=qr_scale, interpolation=cv2.INTER_CUBIC)

# 文字のUnicode値をQRコード化
def create_unicode_qr_code(unicode_value, size=100):
    qr = qrcode.QRCode(version=1, box_size=12, border=1)
    qr.add_data(unicode_value)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    qr_img = cv2.cvtColor(np.array(qr_img), cv2.COLOR_RGB2BGR)
    return cv2.resize(qr_img, (size, size), interpolation=cv2.INTER_CUBIC)

def add_text_and_qr(pil_img, text, font, qr_img, width):
    draw = ImageDraw.Draw(pil_img)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    qr_x = (width - qr_img.shape[1] - text_width - 20) // 2  # テキストの幅を考慮して中央寄せ
    qr_y = 20  # 上部から少し離れるように設定

    pil_img.paste(Image.fromarray(qr_img), (qr_x, qr_y))
    text_x = qr_x + qr_img.shape[1] + 20  # QRコードとテキストの間隔を少し開ける
    text_y = qr_y + (qr_img.shape[0] - text_height) // 2  # QRコードの高さに合わせて中央寄せ
    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))
    return pil_img

def draw_frame(pil_img, width, height, text_font_path, text_characters):
    draw = ImageDraw.Draw(pil_img)
    center_x, center_y = width // 2, height // 2
    frame_width, frame_height = 500 * 5, 500 * 6  # 枠のサイズ

    # 枠の左上と右下の座標
    x1, y1 = center_x - frame_width // 2, center_y - frame_height // 2
    x2, y2 = x1 + frame_width, y1 + frame_height

    # 外枠を描く
    draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], fill=(255, 255, 0), width=1)

    # 横に5分割する線を描く
    for i in range(1, 5):
        line_x = x1 + (frame_width // 5) * i
        draw.line([(line_x, y1), (line_x, y2)], fill=(255, 255, 0), width=1)

    # 縦に6分割する線を描く
    for i in range(1, 6):
        line_y = y1 + (frame_height // 6) * i
        draw.line([(x1, line_y), (x2, line_y)], fill=(255, 255, 0), width=1)

    # テキストフォントとサイズを設定
    text_font = ImageFont.truetype(text_font_path, 60)
    unicode_font = ImageFont.truetype(text_font_path, 20)  # Unicode値のフォントサイズは小さく
    margin = 30
    char_index = 0
    for row in range(6):
        for col in range(5):
            if char_index >= len(text_characters):
                break
            center_x_cell = x1 + (frame_width // 5 * col) + 500 // 2
            center_y_cell = y1 + (frame_height // 6 * row) + 500 // 2
            half_square_size = 250 // 2
            square_x1, square_y1 = center_x_cell - half_square_size, center_y_cell - half_square_size

            # 250pxの正方形を描く
            draw.rectangle(
                [(square_x1, square_y1),
                 (square_x1 + 250, square_y1 + 250)],
                outline=(255, 255, 0),  # 黄色の枠
                width=1
            )

            # 正方形内を縦と横に3等分する点線を追加
            line_length = 1  # 点線の長さ
            gap_length = 7    # 点線の間隔

            # 縦方向
            for i in range(1, 3):  # 2つの点線を引く
                line_x = square_x1 + (250 // 3) * i
                for j in range(0, 250, line_length + gap_length):
                    draw.line([(line_x, square_y1 + j), (line_x, square_y1 + j + line_length)], fill=(255, 255, 0), width=1)

            # 横方向
            for i in range(1, 3):  # 2つの点線を引く
                line_y = square_y1 + (250 // 3) * i
                for j in range(0, 250, line_length + gap_length):
                    draw.line([(square_x1 + j, line_y), (square_x1 + j + line_length, line_y)], fill=(255, 255, 0), width=1)

            # 中心に小さな十字を追加
            cross_size = 15  # 十字のサイズ
            center_x_cross = square_x1 + 250 // 2
            center_y_cross = square_y1 + 250 // 2
            draw.line([(center_x_cross - cross_size, center_y_cross),
                       (center_x_cross + cross_size, center_y_cross)],
                      fill=(255, 255, 0), width=1)
            draw.line([(center_x_cross, center_y_cross - cross_size),
                       (center_x_cross, center_y_cross + cross_size)],
                      fill=(255, 255, 0), width=1)

            # 文字を正方形の左上に表示
            text = text_characters[char_index]
            text_bbox = draw.textbbox((0, 0), text, font=text_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # 文字を正方形の左上に配置
            text_x = square_x1 - text_width - margin
            text_y = square_y1 - text_height - margin
            draw.text((text_x, text_y), text, font=text_font, fill=(0, 0, 0))

            # Unicode値をQRコードとして正方形の外側の右上に表示
            unicode_value = f"U+{ord(text):04X}"
            qr_code = create_unicode_qr_code(unicode_value, size=int(110))  
            qr_margin = 5  # QRコードと正方形の間にマージンを追加
            qr_x = square_x1 + 250 + qr_margin  # 正方形の右端+マージン
            qr_y = square_y1 - qr_code.shape[0] - qr_margin  # 正方形の上端-QRコードの高さ-マージン
            pil_img.paste(Image.fromarray(qr_code), (qr_x, qr_y))

            # Unicode値を正方形の下に表示
            unicode_value = f"U+{ord(text):04X}"  # Unicode値をU+XXXXの形式で表示
            unicode_font = ImageFont.truetype(text_font_path, 45)
            unicode_bbox = draw.textbbox((0, 0), unicode_value, font=unicode_font) 
            unicode_width = unicode_bbox[2] - unicode_bbox[0] 
            unicode_height = unicode_bbox[3] - unicode_bbox[1] 
            unicode_x = square_x1 + (250 - unicode_width) // 2  # 中央揃え 
            unicode_y = square_y1 + 250 + 10  # 正方形の下に少し余裕を持たせて 
            draw.text((unicode_x, unicode_y), unicode_value, font=unicode_font, fill=(0, 0, 0))             

            char_index += 1
        else:
            continue
        break  # 文字が足りなくなったら外側のループも終了

    return pil_img

def add_aruco_markers(img, width, height):
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    marker_size = int(min(width, height) * 0.06)
    positions = [
        (0, 0), (width - marker_size, 0),
        (0, height - marker_size), (width - marker_size, height - marker_size)
    ]
    for idx, pos in enumerate(positions, start=1):
        ar_img = aruco.generateImageMarker(dictionary, idx, marker_size)
        img[pos[1]:pos[1] + marker_size, pos[0]:pos[0] + marker_size] = cv2.cvtColor(ar_img, cv2.COLOR_GRAY2BGR)
    return img

def create_images_with_text_from_file(output_folder, font_path, text_font_path):  
    # 出力フォルダが存在しない場合は作成  
    if not os.path.exists(output_folder):  
        os.makedirs(output_folder)  
  
    width, height = setup_image_dimensions()  
    font_size = int(height * 0.02)  # フォントサイズをA4の2%に設定  
    font = ImageFont.truetype(font_path, font_size)  
  
    # font_settings.txtから文字を読み込み、重複を削除して使用
    with open("font_settings.txt", "r", encoding="utf-8") as file:  
        text_characters = list(file.read().strip())  
    
    # 重複を削除 (最初の出現のみ残す)
    seen = set()
    text_characters = [char for char in text_characters if char not in seen and not seen.add(char)]
  
    # 必要な画像の枚数を計算  
    num_images = (len(text_characters) + 29) // 30  # 1枚あたり30文字分を考慮  
  
    for i in range(num_images):  
        img = create_blank_image(width, height)  
        # ファイル名をゼロ埋めした形式に変更 
        file_id = f"{i+1:03}"  # 3桁のゼロ埋め  
        qr_img = generate_qr_code(f"blank_form_{file_id}", min(width, height) * 0.1, width, height)  
        pil_img = Image.fromarray(img)  
        pil_img = add_text_and_qr(pil_img, f"blank_form_{file_id}", font, qr_img, width)  
  
        # 現在の画像に割り当てる文字のスライスを取る  
        start_char = i * 30  
        end_char = start_char + 30  
        chars_for_this_image = text_characters[start_char:end_char]  
  
        pil_img = draw_frame(pil_img, width, height, text_font_path, chars_for_this_image)  
        img = np.array(pil_img)  
        img = add_aruco_markers(img, width, height)  

        border_size = 50  # 拡張するピクセル数

        # NumPy配列をPillow画像に変換してから拡張
        img = Image.fromarray(img)
        img = ImageOps.expand(img, border=border_size, fill="white")

        # 必要に応じてNumPy配列に戻す
        img = np.array(img)
     
        # 画像を出力する  
        filename = os.path.join(output_folder, f"blank_form_{file_id}.png")  
        cv2.imwrite(filename, img)  
        print(f"blank_form_{file_id}.png を出力しました")

# 実行
output_folder = "01_output_images"
font_path = r"C:\your\font.ttf"
text_font_path = r"C:\your\font.ttf"
create_images_with_text_from_file(output_folder, font_path, text_font_path)
