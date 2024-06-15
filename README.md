## セットアップ

```bash
$ git clone https://github.com/matsuda-tkm/draw-reversi.git
$ cd draw-reversi
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install --upgrade pip
$ pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu117
$ pip install -r requirements.txt
```

## Egaroucid棋譜データのダウンロード

[Egaroucidのサイト](https://www.egaroucid.nyanyan.dev/ja/technology/transcript/)で山名琢翔さんが公開しているデータを使用

```bash
$ mkdir data
$ cd data
$ wget https://github.com/Nyanyan/Egaroucid/releases/download/transcript/Egaroucid_Transcript.zip
$ unzip Egaroucid_Transcript.zip
```

## 学習

`src/train/config.py`に学習の設定を記述

```python
config = {
    "hidden_channels": 4,
    "conv_layers": 2,
    "optimizer": "Adam",
    "lr": 0.01,
    "scheduler": "None",
    "criterion": "HuberLoss",
    "batch_size": 8192,
    "seed": 42,
    "augmentation": True,
    "n_epoch": 10,
    "n_data": 20
}
```

以下のコマンドで学習を実行

```bash
$ cd draw-reversi
$ python -m src.train.main
```
