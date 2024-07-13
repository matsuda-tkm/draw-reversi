import copy

import creversi

from src.search.evaluate import evaluate_func

cache = {}


def alphabeta(node: creversi.Board, depth: int, alpha: float, beta: float, my_turn: bool) -> float:
    global cache

    # ゲームの状態を文字列としてキャッシュのキーにする
    key = (node.to_line(), depth, my_turn)

    # キャッシュにあればそれを返す
    if key in cache:
        return cache[key]

    elif node.is_game_over() or depth == 0:
        score = evaluate_func(node)
        cache[key] = score
        return score

    elif node.turn != my_turn:
        value = float("-inf")
        for move in node.legal_moves:
            child = copy.copy(node)
            child.move(move)
            value = max(value, alphabeta(child, depth - 1, alpha, beta, my_turn))
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        cache[key] = value
        return alpha
    else:
        value = float("inf")
        for move in node.legal_moves:
            child = copy.copy(node)
            child.move(move)
            value = min(value, alphabeta(child, depth - 1, alpha, beta, my_turn))
            beta = min(beta, value)
            if alpha >= beta:
                break

        cache[key] = value
        return beta


if __name__ == "__main__":
    # import random
    import time

    board = creversi.Board()
    print(board)
    depth = 3
    boundary = 54

    start = time.perf_counter()
    while not board.is_game_over():
        # AIの手番
        if board.turn:
            best_move = None
            best_score = float("inf")
            for move in board.legal_moves:
                child = copy.copy(board)
                child.move(move)
                score = alphabeta(child, depth, -float("inf"), float("inf"), board.turn)
                if score < best_score:
                    best_move = move
                    best_score = score
            print(f"best move: {creversi.move_to_str(best_move)}, score: {best_score * 64}")
            board.move(best_move)

        # ランダムプレイヤーの手番
        else:
            move = list(board.legal_moves)[0]
            # move = random.choice(list(board.legal_moves))
            board.move(int(move))

        # 深さ切り替え
        if board.piece_sum() >= boundary and depth != 100:
            print(f"depth change: {depth} --> ∞")
            depth = 100
        print(board)

    print(f"diff: {board.diff_num()}")
    print(f"elapsed time: {time.perf_counter() - start} [s]")
