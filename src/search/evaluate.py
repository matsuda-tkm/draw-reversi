from typing import List

import creversi
import torch
import torch.ao.quantization

from src.train.model import ResidualValueNetwork
from src.utils.boards import BoardCoverter

WEIGHT_PATH = "checkpoints/checkpoint_018.pth"
DEVICE = torch.device("cpu")
MODEL = ResidualValueNetwork(hidden_channels=8, conv_layers=7)
MODEL.load_state_dict(torch.load(WEIGHT_PATH))
MODEL.eval()
MODEL.to(DEVICE)


def evaluate_func(board: creversi.Board) -> float:
    if board.is_game_over():
        return abs(board.diff_num() / 64)
    else:
        board_array = BoardCoverter().to_torch(board).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            value = MODEL(board_array).item()
        return abs(value)


def evaluate_batch(boards: List[creversi.Board]) -> List[float]:
    board_arrays = [BoardCoverter().to_torch(board) for board in boards]
    with torch.no_grad():
        values = MODEL(torch.stack(board_arrays).to(DEVICE)).squeeze()
    values = abs(values.cpu().numpy()).tolist()
    if not isinstance(values, list):
        values = [abs(values)]
    return values


if __name__ == "__main__":
    import time

    # 推論速度を測定
    N = 1000

    # 1バッチでの推論時間
    start = time.time()
    for _ in range(N):
        board = creversi.Board()
        evaluate_func(board)
    single_time = time.time() - start
    print(f"single batch: {single_time} [s]")

    # バッチ処理での推論時間
    board_list = [creversi.Board() for _ in range(N)]
    start = time.time()
    evaluate_batch(board_list)
    batch_time = time.time() - start
    print(f"batch: {batch_time} [s]")

    print(f"speedup: {single_time / batch_time}")
