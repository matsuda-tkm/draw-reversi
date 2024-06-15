from typing import Callable, Tuple, Optional

import creversi
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class ScriptDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, dtype: torch.dtype = torch.float32):
        self.X = X
        self.y = y
        self.dtype = dtype

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_tensor = torch.from_numpy(self.X[idx]).to(self.dtype)
        y_tensor = torch.from_numpy(self.y[idx]).to(self.dtype) / 64
        return X_tensor, y_tensor


def load_data(
        path_to_script: Path,
        board_converter: Callable[[creversi.Board], np.ndarray],
        read_lines: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    with open(path_to_script, "r") as f:
        transcripts = f.read().split("\n")[:-1]
        if read_lines is not None:
            transcripts = transcripts[:read_lines]
    for transcript in transcripts:
        board = creversi.Board()
        board_arrays = []
        for i in range(0, len(transcript), 2):
            move = creversi.move_from_str(transcript[i: i + 2])
            board.move(move)
            if i > 19:
                board_arrays.append(board_converter(board))
        if len(board_arrays) == 0:
            continue
        z = board.diff_num() if board.turn else -board.diff_num()
        board_arrays = np.array(board_arrays)
        if np.ndim(board_arrays) == 5:
            board_arrays = np.concatenate(board_arrays)
        y += [z] * board_arrays.shape[0]
        x += [board_arrays]
    x = np.concatenate(x).astype(np.float16)
    y = np.array(y, dtype=np.int8).reshape(-1, 1)
    return x, y


if __name__ == "__main__":
    from src.train.dataset_path import egaroucid_dir
    from src.utils.boards import BoardCoverter

    board_converter = BoardCoverter()
    for path_to_script in egaroucid_dir.iterdir():
        x, y = load_data(str(path_to_script), board_converter.to_numpy)
        print(x.shape, y.shape)
        break
