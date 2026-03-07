## Rust高速シミュレータ（探索用）

学習中/推論時に探索（Expectimax等）を行うための高速盤面シミュレータを `rust_sim/` に追加しています。

### ビルド

```bash
cd rust_sim
maturin develop --release
```

### 簡易動作確認

```bash
uv run python rust_sim_smoke_test.py
```

### Pythonからの利用

`fast2048.py` のAPIを利用してください。

- `apply_action(board, action) -> (next_board, reward, moved)`
- `legal_actions(board) -> list[int]`
- `empty_positions(board) -> list[int]`
- `spawn_tile(board, position, value_exp) -> list[int]`
