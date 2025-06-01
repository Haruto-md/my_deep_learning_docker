from torch.utils.data import Dataset

class APTOSDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = load_image(row["image_path"])  # 自前で定義
        label = row["label"]

        if self.transform:
            image = self.transform(image)

        return image, label