from dotenv import load_dotenv

from loguru import logger
import numpy as np
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

from src.train.model import ValueNetwork
from src.train.dataset import ScriptDataset
from src.train.config import config

assert torch.cuda.is_available(), "CUDA is not available"
load_dotenv()


class Trainer:
    def __init__(self) -> None:
        self.config = config

    def set_model(self):
        self.model = ValueNetwork(
            hidden_channels=self.config["hidden_channels"],
            conv_layers=self.config["conv_layers"]
        )
        self.model = nn.DataParallel(self.model)
        self.model.to("cuda")

    def set_optimizer(self):
        if self.config["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["lr"]
            )
        else:
            raise ValueError(f"Invalid optimizer: {self.config['optimizer']}")

    def set_scheduler(self):
        if self.config["scheduler"] == "None":
            pass
        elif self.config["scheduler"] == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config["gamma"]
            )
        else:
            raise ValueError(f"Invalid scheduler: {self.config['scheduler']}")

    def set_criterion(self):
        if self.config["criterion"] == "HuberLoss":
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Invalid criterion: {self.config['criterion']}")

    def make_dataloader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        dataloader = DataLoader(
            ScriptDataset(X, y),
            batch_size=self.config["batch_size"],
            shuffle=shuffle
        )
        return dataloader

    def seed_everything(self) -> None:
        import random
        import os
        import numpy as np
        seed = self.config["seed"]
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def train(self, train_dataloader: DataLoader, eval_dataloader: DataLoader) -> None:
        for epoch in range(config["n_epoch"]):
            train_loss, n = 0, 0
            eval_loss, m = 0, 0
            for X, y in train_dataloader:
                X, y = X.to("cuda"), y.to("cuda")
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                n += 1
            if self.config["scheduler"] != "None":
                self.scheduler.step()
            train_loss /= n
            self.model.eval()
            for X, y in eval_dataloader:
                X, y = X.to("cuda"), y.to("cuda")
                with torch.no_grad():
                    output = self.model(X)
                loss = self.criterion(output, y)
                eval_loss += loss.item()
                m += 1
            eval_loss /= m
            self.model.train()
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "learning rate": self.optimizer.param_groups[0]["lr"]
                })
            logger.info(f"epoch: {epoch}, train loss: {train_loss}, eval loss: {eval_loss}")

    def save_model(self, path: Path) -> None:
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        torch.save(self.model.module.state_dict(), path)
        if wandb.run is not None:
            wandb.save(str(path))

    def start_wandb(self, run_name: str) -> None:
        wandb.init(project="draw-othello", name=run_name, config=self.config)

    def finish_wandb(self, text: str) -> None:
        if wandb.run is not None:
            wandb.alert(title="Finish training", text=text, level="INFO")
            wandb.finish()
        else:
            logger.warning("wandb is not started")


if __name__ == "__main__":
    from src.train.dataset_path import egaroucid_dir
    from src.train.dataset import load_data
    from src.utils.boards import BoardCoverter

    logger.info("Start running")
    trainer = Trainer()
    trainer.seed_everything()
    trainer.set_model()
    trainer.set_optimizer()
    trainer.set_scheduler()
    trainer.set_criterion()
    if trainer.config["augmentation"]:
        board_converter = BoardCoverter().to_numpy_augmented
    else:
        board_converter = BoardCoverter().to_numpy
    generator = sorted(egaroucid_dir.iterdir())
    for i, path_to_script in enumerate(generator[:trainer.config["n_data"] + 1]):
        if i == 0:
            x, y = load_data(str(path_to_script), board_converter)
            eval_dataloader = trainer.make_dataloader(x, y, shuffle=False)
            logger.success("Eval dataloader is created")
            continue
        logger.info(f"loading {i}th data")
        x, y = load_data(str(path_to_script), board_converter)
        logger.debug(x.shape)
        train_dataloader = trainer.make_dataloader(x, y, shuffle=True)
        if i == 1:
            trainer.start_wandb("10epoch, Augmented, 0.01, 8192, h=4, l=2")
            pass
        trainer.train(train_dataloader, eval_dataloader)
        trainer.save_model(Path(f"checkpoints/checkpoint_{i:03}.pth"))
    trainer.finish_wandb("引き分けオセロAIの学習が終了しました")
