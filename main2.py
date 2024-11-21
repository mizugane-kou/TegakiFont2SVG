import cv2 
import numpy as np
from cv2 import aruco
import os
import glob
from PIL import Image, ImageEnhance
from pyzbar.pyzbar import decode
import shutil
import svgwrite
from pathlib import Path
from scipy.interpolate import splprep, splev

def clear_folder(folder_path):
    """指定したフォルダ内の全ファイルを削除する"""
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

def add_border(img, border_ratio):
    # 画像のサイズを取得
    height, width = img.shape[:2]
    # 余白のサイズを計算
    top = int(height * border_ratio)
    bottom = int(height * border_ratio)
    left = int(width * border_ratio)
    right = int(width * border_ratio)
    # 白い余白を追加して新しい画像を作成
    img_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img_with_border

def sanitize_filename(filename):
    # ファイル名に使用できない文字を置き換え
    return "".join(c if c.isalnum() or c in ('-', '_') else "_" for c in filename)

def correct_distortion(input_file, output_folder, border_ratio=0.1): 
    # ファイルパスを正しく処理 
    input_file = os.path.normpath(input_file)  # Windowsパスを適切に処理 
    input_file = input_file.encode('utf-8').decode('utf-8')  # Unicodeとしてデコード 
 
    # 画像を読み込む 
    img = cv2.imread(input_file) 
 
    # 画像が正しく読み込まれなかった場合のエラーチェック 
    if img is None: 
        print(f"Error: {input_file} の画像を読み込めませんでした。ファイルパスが正しいか確認してください。") 
        return 
     
    # 画像に余白を追加 
    img_with_border = add_border(img, border_ratio) 
 
    # グレイスケールに変換 
    gray = cv2.cvtColor(img_with_border, cv2.COLOR_BGR2GRAY) 
 
    # pyzbarでQRコードの検出
    decoded_objects = decode(gray)

    # 有効なQRコードが見つかったかどうかのフラグ
    valid_qr_found = False
    qr_data = None

    # 全てのQRコードを読み出し、有効かどうかを確認
    for obj in decoded_objects:
        qr_data = obj.data.decode('utf-8')
        if qr_data.startswith("blank_form"):
            valid_qr_found = True
            # QRコードデータが重複していないかチェック 
            if hasattr(correct_distortion, "qr_data_set"): 
                if qr_data in correct_distortion.qr_data_set: 
                    print(f"Error: 重複したQRコードデータが検出されました: {qr_data}") 
                    return 
                correct_distortion.qr_data_set.add(qr_data) 

            # ファイル名をQRコードのデータから作成 
            filename = sanitize_filename(qr_data) + ".png"
            break  # 最初に見つかった有効なQRコードを使用

    if not valid_qr_found:
        print(f"Warning: {input_file} に有効なQRコードが見つかりませんでした")
        filename = "corrected_" + os.path.basename(input_file)

    output_file = os.path.join(output_folder, filename)

    # ArUcoマーカーの検出
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    
    if len(corners) != 4:
        print(f"Error: {input_file} に正確に4つのArUcoマーカーが見つかりませんでした")
        return

    # 画像サイズ
    img_height, img_width = img_with_border.shape[:2]
    
    # A4サイズ (210mm x 297mm) に対応するピクセル数
    width, height = 2893, 4092  # A4サイズ
    
    # マーカーのサイズ
    marker_size = int(min(width, height) * 0.06)
    
    # A4用の四隅座標を設定
    src_points = np.float32([[0, 0], [width - marker_size, 0],
                             [0, height - marker_size], [width - marker_size, height - marker_size]])
    
    # ID順に並べ替える
    ordered_corners = [None] * 4
    for i, marker_id in enumerate(ids.flatten()):
        ordered_corners[marker_id - 1] = corners[i][0][0]
    
    # 変換行列を計算
    dst_points = np.float32(ordered_corners)
    M = cv2.getPerspectiveTransform(dst_points, src_points)
    
    # 歪み補正を適用
    corrected = cv2.warpPerspective(img_with_border, M, (width, height))
    
    # 補正された画像を保存
    cv2.imwrite(output_file, corrected)
    print(f"{output_file} に保存されました")

# QRコードデータのセットを保持（重複チェック用）
correct_distortion.qr_data_set = set()

def process_images_in_folder(input_folder, output_folder, border_ratio=0.1):
    # 出力フォルダが存在しない場合、作成
    os.makedirs(output_folder, exist_ok=True)
    
    # 出力フォルダを空にする（関数 clear_folder を実装済みと仮定）
    clear_folder(output_folder)
    
    # 指定したフォルダ内の全ての画像ファイル（.png と .jpg）を取得
    image_files = glob.glob(os.path.join(input_folder, "*.png")) + \
                  glob.glob(os.path.join(input_folder, "*.jpg"))
    
    # 各画像ファイルに対して処理を実行
    for input_file in image_files:
        # 歪み補正を実行（関数 correct_distortion を実装済みと仮定）
        correct_distortion(input_file, output_folder, border_ratio)

def process_images(input_dir, output_dir, crop_width=2500, crop_height=3000, num_rows=6, num_cols=5):
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    # 出力フォルダを空にする
    clear_folder(output_dir)

    # 各セルの幅と高さを計算
    cell_width = crop_width // num_cols
    cell_height = crop_height // num_rows

    # 入力ディレクトリ内の画像を処理
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 画像の読み込み
            input_path = os.path.join(input_dir, filename)
            with Image.open(input_path) as img:
                # 画像の中央部を切り取り
                img_width, img_height = img.size
                left = (img_width - crop_width) // 2
                upper = (img_height - crop_height) // 2
                cropped_img = img.crop((left, upper, left + crop_width, upper + crop_height))

                # 切り取った部分を分割
                base_name, ext = os.path.splitext(filename)
                for row in range(num_rows):
                    for col in range(num_cols):
                        # 各セルの座標計算
                        cell_left = col * cell_width
                        cell_upper = row * cell_height
                        cell_right = cell_left + cell_width
                        cell_lower = cell_upper + cell_height
                        cell_img = cropped_img.crop((cell_left, cell_upper, cell_right, cell_lower))

                        # ファイル名を生成
                        cell_number = row * num_cols + col + 1
                        output_filename = f"{base_name}_{cell_number:02}{ext}"
                        output_path = os.path.join(output_dir, output_filename)

                        # 分割した画像を保存
                        cell_img.save(output_path)

    print("分割処理１が完了しました")

def adjust_image_properties(image, brightness=0, contrast=1.2, saturation=0):
    # Adjust brightness and contrast
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    
    # Convert to HSV to adjust saturation
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.convertScaleAbs(s, alpha=saturation)
    hsv_adjusted = cv2.merge([h, s, v])
    adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    
    return adjusted

def process_qr_images(input_folder, output_folder, brightness=0.1, contrast=1.2, saturation=0):
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    # 出力フォルダを空にする
    clear_folder(output_folder)
    
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if not os.path.isfile(file_path):
            continue  # Skip non-file entries

        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Skipping invalid image: {file_name}")
            continue

        # Decode QR codes
        qr_codes = decode(image)
        if not qr_codes:
            print(f"No QR code found in: {file_name}")
            continue

        # Use the first QR code's data as the new filename
        qr_name = qr_codes[0].data.decode('utf-8')

        # Crop the central 250x250 px area of the image
        height, width, _ = image.shape
        center_x, center_y = width // 2, height // 2
        crop_size = 250
        half_crop = crop_size // 2

        # Ensure cropping dimensions are within the image boundaries
        x1 = max(center_x - half_crop, 0)
        y1 = max(center_y - half_crop, 0)
        x2 = min(center_x + half_crop, width)
        y2 = min(center_y + half_crop, height)

        cropped_image = image[y1:y2, x1:x2]

        # Adjust image properties
        adjusted_image = adjust_image_properties(cropped_image, brightness, contrast, saturation)

        blurred_image = cv2.GaussianBlur(adjusted_image, (5, 5), 0)

        # OpenCVの画像をPillow形式に変換
        blurred_image_pil = Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))

        # コントラストを上げる
        enhancer = ImageEnhance.Contrast(blurred_image_pil)
        enhanced_image_pil = enhancer.enhance(5)  # コントラストの強度を調整      

        enhanced_image_cv = cv2.cvtColor(np.array(enhanced_image_pil), cv2.COLOR_RGB2BGR) 

        uniname = replace_uplus_with_uni(qr_name) 

        # Save the adjusted image with the new name
        output_path = os.path.join(output_folder, f"{uniname}.png")
        cv2.imwrite(output_path, enhanced_image_cv)
        #print(f"Processed: {file_name} -> {qr_name}.png")
        
    print("分割処理２が完了しました")


def smooth_contour(contour, smoothing_factor=0.1):
    """
    隣接する頂点の法線方向で微調整してスムージングを行う。

    Args:
        contour (numpy.ndarray): スムージング対象の輪郭。
        smoothing_factor (float): スムージングの強さ（小さいほど効果が弱い）。

    Returns:
        numpy.ndarray: スムージングされた輪郭。
    """
    # (N, 1, 2) -> (N, 2) に変換
    contour = contour[:, 0, :]

    smoothed_contour = []
    for i in range(len(contour)):
        prev_point = contour[i - 1] if i > 0 else contour[-1]
        next_point = contour[i + 1] if i < len(contour) - 1 else contour[0]

        # 隣接点とのベクトル差を計算
        prev_vector = prev_point - contour[i]
        next_vector = next_point - contour[i]

        # ベクトルの法線方向を計算
        normal = (prev_vector + next_vector) / 2
        normal /= np.linalg.norm(normal)  # 正規化

        # スムージングするために法線方向にわずかに調整
        smoothed_point = contour[i] + smoothing_factor * normal
        smoothed_contour.append(smoothed_point)

    return np.array(smoothed_contour)



def extract_black_regions_to_svg(input_folder, output_folder):  
    """  
    指定されたフォルダ内の画像から黒い部分を抽出してSVGに変換し、出力フォルダに保存します。  
  
    Args:  
        input_folder (str): 入力画像フォルダのパス。  
        output_folder (str): 出力SVGフォルダのパス。  
    """  
    # 入力フォルダと出力フォルダのパスを作成  
    input_path = Path(input_folder)  
    output_path = Path(output_folder)  
  
    # 出力フォルダが存在しない場合は作成  
    output_path.mkdir(parents=True, exist_ok=True)  
  
    # 対応する画像ファイルの拡張子  
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']  
  
    # 入力フォルダ内のすべての画像ファイルを処理  
    for file in input_path.iterdir():  
        if file.suffix.lower() in valid_extensions:  
            # 画像を読み込む  
            image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)  
            original_height, original_width = image.shape

            # 画像を拡大  
            image = cv2.resize(image, (original_width * 4, original_height * 4), interpolation=cv2.INTER_LINEAR)  
            height, width = image.shape  

            # 画像を2値化（黒い部分を抽出）  
            _, binary_image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY_INV)  
  
            # 輪郭を抽出（RETR_TREEを使用して内側の輪郭も取得）  
            contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
  
            # SVGの作成（拡大した画像のサイズに基づいて設定）  
            svg_filename = output_path / f"{file.stem}.svg"  
            dwg = svgwrite.Drawing(  
                str(svg_filename),  
                size=(f"{original_width}px", f"{original_height}px"),  # viewBoxは元の画像サイズに合わせる  
                viewBox=f"0 0 {original_width} {original_height}"  # viewBoxも元のサイズに戻す
            )  

            # SVGのサイズ縮小倍率（例: 0.25倍）  
            scale_factor = 0.25  

            # 輪郭が見つからない場合は空のSVGを生成  
            if not contours:  
                # 真っ白な背景のSVGを生成  
                dwg.add(dwg.rect(insert=(0, 0), size=(original_width, original_height), fill="white"))  
            else:  
                # 輪郭をSVGの1つのパスに統合して描画（塗りつぶしとくり抜き）  
                path_data = []  
                for contour in contours:  
                    # 輪郭を滑らかにする（頂点の数を増やす） 
                    epsilon = 1.5 
                    contour = cv2.approxPolyDP(contour, epsilon, True) 
                    contour = smooth_contour(contour, smoothing_factor=1) 
 
                    # 座標リストを適切な形式に変換（整数を浮動小数点にする） 
                    points = [f"{x * scale_factor:.2f},{y * scale_factor:.2f}" for x, y in contour]  # 座標を縮小
                    path_data.append("M " + " L ".join(points) + " Z") 
 
                # パスを1つに統合（外側と内側のパスを組み合わせる） 
                path_data = " ".join(path_data) 
 
                # SVGに追加（塗りつぶしとくり抜き）  
                dwg.add(dwg.path(d=path_data, fill="black", fill_rule="evenodd", stroke="none")) 

            # SVGを保存  
            dwg.save()  
    print("SVGの出力が完了しました")

def replace_uplus_with_uni(input_string):
    return input_string.replace("U+", "uni")



# スキャンした画像を歪み補正 名前を正規化
process_images_in_folder("02_input_images", "03_output_images", border_ratio=0.1)

# 中央部から枠内の画像を切り出して分割
process_images("03_output_images", "04_output_images")

# Unicode値を取得してグリフ画像のみを出力
process_qr_images("04_output_images", "05_output_images", brightness=0, contrast=1.4, saturation=0)

# SVGに変換
extract_black_regions_to_svg("05_output_images", "06_output_images")



