from fast2048 import apply_action, empty_positions, is_available, legal_actions, spawn_tile


def main() -> None:
    if not is_available():
        print("fast2048_sim is not available. Build with: cd rust_sim && maturin develop --release")
        return

    board = [0] * 16
    board[0] = 1
    board[1] = 1

    next_board, reward, moved = apply_action(board, 3)  # LEFT
    print("moved:", moved)
    print("reward:", reward)
    print("next_board:", next_board)
    print("legal_actions:", legal_actions(next_board))
    print("empty_count:", len(empty_positions(next_board)))

    empties = empty_positions(next_board)
    spawned = spawn_tile(next_board, empties[0], 1)
    print("spawned_first_empty:", spawned)


if __name__ == "__main__":
    main()