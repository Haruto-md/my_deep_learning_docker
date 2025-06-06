import os
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import clip
import pandas as pd

# CLIPモデル読み込み
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# 単一動画から特徴抽出
def extract_clip_features_fixed_fps(video_path, target_fps=15, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps == 0:
        orig_fps = 30  # fallback

    interval = int(round(orig_fps / target_fps))
    features = []
    frame_count = 0
    sampled = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            image_input = preprocess(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features = image_features.cpu().numpy().squeeze()
                features.append(image_features)

            sampled += 1
            if max_frames and sampled >= max_frames:
                break

        frame_count += 1

    cap.release()
    return np.stack(features) if features else None


# バッチ処理関数
def process_video_folder(input_dir, output_dir, target_fps=15, max_frames=None):
    os.makedirs(output_dir, exist_ok=True)
    video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')

    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))

    print(f"{len(video_files)} videos found in {input_dir}")

    for video_path in tqdm(video_files):
        try:
            filename = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_dir, f"{filename}.npy")

            if os.path.exists(output_path):
                continue  # 既に処理済み

            features = extract_clip_features_fixed_fps(video_path, target_fps, max_frames)
            if features is not None:
                np.save(output_path, features)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

def generate_groundtruth_from_csv(
    annotation_csv,
    feature_dir,
    output_dir,
    target_fps=15,
    label_column="phase_id",
    background_label="17"
):
    os.makedirs(output_dir, exist_ok=True)

    # 全アノテーション読み込み
    df = pd.read_csv(annotation_csv)

    # 各動画ごとに処理
    video_ids = df['video_id'].unique()

    for vid in tqdm(video_ids):
        feature_path = os.path.join(feature_dir, f"{vid}.npy")
        output_path = os.path.join(output_dir, f"{vid}.txt")

        if not os.path.exists(feature_path):
            print(f"Warning: feature not found for {vid}")
            continue

        features = np.load(feature_path)
        num_frames = features.shape[0]
        frame_times = np.arange(num_frames) / target_fps

        # 対応するラベルデータを抽出
        df_vid = df[df['video_id'] == vid]

        # ラベル付け
        labels = []
        for t in frame_times:
            row = df_vid[(df_vid['start'] <= t) & (df_vid['end'] > t)]
            if len(row) > 0:
                label = str(row.iloc[0][label_column])
            else:
                label = background_label
            labels.append(label)

        # 書き出し
        with open(output_path, "w") as f:
            for label in labels:
                f.write(f"{label}\n")

        print(f"{vid}: {num_frames} frames written to {output_path}")

# 実行設定
input_video_folder = "input/APTOS2025/videos"
output_feature_folder = "FACT/data/APTOS2025/features"
annotation_path = "input\APTOS2025\APTOS_train-val_annotation.csv"
label_path = "FACT\data\APTOS2025\groundTruth"
process_video_folder(input_video_folder, output_feature_folder, target_fps=15)
generate_groundtruth_from_csv(output_feature_folder,annotation_path,label_path,target_fps=15)

import pandas as pd
from sklearn.model_selection import KFold

# アノテーションファイルからユニークな video_id を抽出
df = pd.read_csv(annotation_path)
video_ids = df["video_id"].unique()

# 5-fold CV の準備
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(video_ids), 1):
    train_ids = video_ids[train_idx]
    test_ids = video_ids[test_idx]
    
    # 書き出し
    with open(f"FACT/data/APTOS2025/splits/train.split{fold}.bundle", "w") as f:
        f.writelines(f"{vid}\n" for vid in train_ids)
    with open(f"FACT/data/APTOS2025/splits/test.split{fold}.bundle", "w") as f:
        f.writelines(f"{vid}\n" for vid in test_ids)
