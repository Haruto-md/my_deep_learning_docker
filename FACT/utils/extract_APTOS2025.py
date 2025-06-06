import os
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import clip

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

def generate_frame_labels_from_csv(feature_dir, annotation_dir, output_dir, target_fps=15):
    os.makedirs(output_dir, exist_ok=True)

    feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
    
    for fname in tqdm(feature_files):
        video_id = os.path.splitext(fname)[0]
        feature_path = os.path.join(feature_dir, fname)
        csv_path = os.path.join(annotation_dir, f"{video_id}.csv")
        out_path = os.path.join(output_dir, f"{video_id}.txt")

        # 1. フレーム数を取得
        features = np.load(feature_path)
        num_frames = features.shape[0]

        # 2. 各フレームに対応する秒数を計算
        frame_times = np.arange(num_frames) / target_fps  # 秒単位

        # 3. アノテーションCSVを読み込む（start, end, label）
        df = pd.read_csv(csv_path)
        labels = []

        for t in frame_times:
            matched = df[(df['start'] <= t) & (df['end'] > t)]
            if len(matched) > 0:
                labels.append(matched.iloc[0]['label'])
            else:
                labels.append('background')  # or unknown class

        # 4. 書き出し
        with open(out_path, 'w') as f:
            for label in labels:
                f.write(f"{label}\n")

        # optional log
        print(f"{video_id}: {num_frames} frames → {out_path}")

# 実行設定
input_video_folder = "videos"         # 動画が入っているフォルダ
output_feature_folder = "features"    # .npyを保存するフォルダ
process_video_folder(input_video_folder, output_feature_folder, target_fps=15)
