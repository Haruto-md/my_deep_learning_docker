import pandas as pd
import yaml
from src.trainers.core import Trainer
from src.utils import set_seed

def main():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(cfg["data_csv_path"])
    set_seed(cfg["seed"])
    trainer = Trainer(cfg, df)
    trainer.run()

if __name__ == "__main__":
    main()