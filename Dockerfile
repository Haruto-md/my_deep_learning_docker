FROM gcr.io/kaggle-gpu-images/python:latest

#言語と地域の設定
ENV lang="ja_jp.utf-8" language="ja_jp:ja" lc_all="ja_jp.utf-8"

#ライブラリのインストール
WORKDIR /workspace
#各々のGPUに対応するpytorchをインストールhttps://pytorch.org/get-started/previous-versions/
# RUN pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
ADD requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

#jupyter notebookの起動
ADD run.sh /opt/run.sh
RUN chmod 700 /opt/run.sh
CMD /opt/run.sh