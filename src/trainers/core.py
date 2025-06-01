import torch
import wandb
from sklearn.model_selection import StratifiedKFold
import wandb.util
from src.utils import save_model, AverageMeter, evaluate
from src.datasets.core import APTOSDataset
from torch.utils.data import DataLoader
from src.models.core import get_model

class Trainer:
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.df = df
        # wandb configにはシリアライズ可能な辞書のみ渡す
        wandb_cfg = {k: v for k, v in cfg.items() if self._is_serializable(v)}
        wandb.init(project=cfg["wandb_project"], config=wandb_cfg)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _is_serializable(self, value):
        try:
            wandb.util.json_dumps_safer(value)
            return True
        except:
            return False

    def _setup_model(self):
        model = get_model(self.cfg).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg["lr"])
        criterion = torch.nn.CrossEntropyLoss()
        return model, optimizer, criterion

    def _get_loaders(self, tr_idx, va_idx,transform=None):
        train_dataset = APTOSDataset(tr_idx, transform=transform)
        valid_dataset = APTOSDataset(va_idx)
        train_loader = DataLoader(train_dataset, batch_size=self.cfg["batch_size"], shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=self.cfg["batch_size"], shuffle=False, num_workers=4)
        return train_loader, valid_loader

    def train_one_fold(self, fold, tr_idx, va_idx):
        train_loader, valid_loader = self._get_loaders(tr_idx, va_idx)
        model, optimizer, criterion = self._setup_model()
        best_score = -float("inf")

        for epoch in range(self.cfg["epochs"]):
            train_loss = self.train_one_epoch(model, train_loader, optimizer, criterion)
            val_score = evaluate(model, valid_loader, self.device)
            wandb.log({
                f"fold{fold}/train_loss": train_loss,
                f"fold{fold}/val_score": val_score,
                "epoch": epoch,
            })
            if val_score > best_score:
                best_score = val_score
                save_model(model, f"best_fold{fold}.pt")

    def train_full(self):
        train_loader = DataLoader(self.df, batch_size=self.cfg["batch_size"], shuffle=True, num_workers=4)
        model, optimizer, criterion = self._setup_model()

        for epoch in range(self.cfg["epochs"]):
            train_loss = self.train_one_epoch(model, train_loader, optimizer, criterion)
            wandb.log({"full/train_loss": train_loss, "epoch": epoch})

        save_model(model, "final_model.pt")

    def run(self):
        if self.cfg["n_folds"] > 1:
            kfold = StratifiedKFold(n_splits=self.cfg["n_folds"], shuffle=True, random_state=self.cfg["seed"])
            for fold, (tr_idx, va_idx) in enumerate(kfold.split(self.df, self.df[self.cfg["target_col"]])):
                print(f"==== Fold {fold} ====")
                self.train_one_fold(fold, tr_idx, va_idx)
        else:
            self.train_full() #全データを使って最終学習する。すでにハイパラとモデルが確定している場合。

    def train_one_epoch(self, model, loader, optimizer, criterion):
        model.train()
        loss_meter = AverageMeter()
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            
            optimizer.step()
            loss_meter.update(loss.item(), x.size(0))
        return loss_meter.avg