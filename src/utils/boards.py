import copy

import creversi
import numpy as np
import torch


class BoardCoverter:
    """board --> ndarray
    0. 黒石の位置
    1. 白石の位置
    2. 空白の位置
    3. 合法手の位置
    4. 返せる石の個数
    5. 隅,X,Cの位置
    6. 1埋め
    7. 黒番=0埋め、白番=1埋め
    """
    def __init__(self):
        pass

    def to_numpy(self, board_obj: creversi.Board) -> np.ndarray:
        board = np.zeros((8, 8, 8), dtype=np.float32)
        board_obj.piece_planes(board)

        # 白番の場合
        if not board_obj.turn:
            board = board[[1, 0, 2, 3, 4, 5, 6, 7], :, :]
            board[7] = 1

        # 空白の位置
        board[2] = np.where(board[0] + board[1] == 1, 0, 1)

        # 合法手の位置, 返せる石の個数
        legal_moves = list(board_obj.legal_moves)
        if legal_moves != [64]:
            n_returns = []
            for move in legal_moves:
                board_obj_ = copy.copy(board_obj)
                n_before = board_obj_.opponent_piece_num()
                board_obj_.move(move)
                n_after = board_obj_.piece_num()
                n_returns.append(n_before - n_after)
            tmp = np.zeros(64)
            tmp[legal_moves] = n_returns
            tmp = tmp.reshape(8, 8)
            board[3] = np.where(tmp > 0, 1, 0)
            board[4] = tmp

        # 隅,X,Cの位置
        board[5] = np.array(
            [
                1., 1., 0., 0., 0., 0., 1., 1.,
                1., 1., 0., 0., 0., 0., 1., 1.,
                0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0.,
                1., 1., 0., 0., 0., 0., 1., 1.,
                1., 1., 0., 0., 0., 0., 1., 1.
            ]
        ).reshape(8, 8)

        # 1埋め
        board[6] = 1

        return board

    def to_torch(self, board_obj: creversi.Board) -> torch.Tensor:
        return torch.from_numpy(self.to_numpy(board_obj))

    def to_numpy_augmented(self, board_obj: creversi.Board) -> np.ndarray:
        board = self.to_numpy(board_obj)
        # original
        board_augmented = [board]
        # flip
        board_augmented.append(np.flip(board, axis=2).copy())
        # rotate
        for k in range(1, 4):
            board_rot = np.rot90(board, k=k, axes=(1, 2)).copy()
            board_augmented.append(board_rot)
            board_augmented.append(np.flip(board_rot, axis=2).copy())
        return np.array(board_augmented)

    def to_torch_augmented(self, board_obj: creversi.Board) -> torch.Tensor:
        return torch.from_numpy(self.to_numpy_augmented(board_obj))
