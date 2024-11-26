# TegakiFont2SVG

手書きでフォントを作成するためのツールです。

```
pip install opencv-python numpy Pillow pyzbar svgwrite qrcode scipy
```

main1.pyを使ってfont_settings.txtで指定したグリフの書き込み用シートが作成できます。  
コード末尾のfont_pathを設定して使ってください。設定したフォントは書き込む文字のサンプルを表示する為に使うので[BizinGothic](https://github.com/yuru7/bizin-gothic)などの収録文字数の多いフォントを使うのがおすすめです。

main2.pyを使って[SVG2FontBuilder](https://github.com/NightFurySL2001/SVG2FontBuilder)で読み込めるSVGを作成できます。  
ペイントソフトで手書き、若しくはスキャナーで取り込んだ書き込みシートを02_input_imagesフォルダに入れて実行してください。
06_output_imagesにSVGが出力されます。

※歪曲収差の補正は実装していません。カメラでなくスキャナでの読み込みがおすすめです。




<img src="img/blank_form_001.png" width="512">

※README書き途中です ぼちぼち追加します
