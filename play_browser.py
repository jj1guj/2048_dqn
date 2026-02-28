"""
ブラウザで2048をプレイするWebアプリ
プレイデータ(experience)を自動保存し、強化学習の学習データとして利用可能
"""

import os
import pickle
import signal
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import gymnasium as gym
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ==================== 設定 ====================
HOST = "0.0.0.0"
PORT = 8080
EXPERIENCE_DIR = "experiences"
# ================================================


class Game2048Server:
    """2048ゲームのバックエンド管理"""

    def __init__(self):
        self.env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0")
        self.current_obs: Optional[np.ndarray] = None
        self.current_info: Optional[dict] = None
        self.episode_experiences: list[dict] = []  # 現在のエピソードのexperience
        self.episode_count = 0
        self.step_count = 0  # エピソード内のステップ数(save時にリセットされない)
        os.makedirs(EXPERIENCE_DIR, exist_ok=True)
        self.reset()

    def reset(self):
        """ゲームをリセット"""
        # 前のエピソードのexperienceを保存
        if self.episode_experiences:
            self._save_experience()
        self.episode_experiences = []
        self.step_count = 0
        self.current_obs, self.current_info = self.env.reset()
        return self._get_state()

    def step(self, action: int):
        """1ステップ実行"""
        assert self.current_obs is not None
        prev_obs = self.current_obs.copy()
        obs, reward, terminated, truncated, info = self.env.step(action)

        # ステップカウント(合法手のみ)
        if info["is_legal"]:
            self.step_count += 1

        # experienceを記録(合法手のみ)
        if info["is_legal"]:
            experience = {
                "state": prev_obs,
                "action": action,
                "reward": reward,
                "next_state": obs,
                "terminated": terminated,
                "info": {
                    "step_score": info["step_score"],
                    "total_score": info["total_score"],
                    "max": info["max"],
                },
            }
            self.episode_experiences.append(experience)

        self.current_obs = obs
        self.current_info = info

        state = self._get_state()
        state["reward"] = float(reward)
        state["terminated"] = terminated
        state["is_legal"] = info["is_legal"]

        # ゲーム終了時に自動保存
        if terminated:
            self._save_experience()
            self.episode_experiences = []

        return state

    def _get_state(self):
        """現在のゲーム状態をJSON用dictで返す"""
        assert self.current_info is not None
        board = self.current_info["board"].astype(np.int32)
        # log2値を実際のタイル値に変換 (0はそのまま0)
        display_board = np.where(board > 0, 2**board, 0)
        return {
            "board": display_board.tolist(),
            "score": int(self.current_info["total_score"]),
            "max_tile": int(2 ** self.current_info["max"]) if self.current_info["max"] > 0 else 0,
            "steps": self.step_count,
        }

    def _save_experience(self):
        """エピソードのexperienceをpickleファイルに保存"""
        if not self.episode_experiences:
            return
        self.episode_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        total_score = self.episode_experiences[-1]["info"]["total_score"]
        max_tile = 2 ** self.episode_experiences[-1]["info"]["max"]
        filename = f"episode_{self.episode_count:04d}_{timestamp}_score{total_score}_max{max_tile}.pkl"
        filepath = os.path.join(EXPERIENCE_DIR, filename)

        # numpy配列をそのまま保存(DQN学習に直接使える形式)
        save_data = {
            "episode": self.episode_count,
            "timestamp": timestamp,
            "total_score": total_score,
            "max_tile": max_tile,
            "num_steps": len(self.episode_experiences),
            "transitions": self.episode_experiences,
        }
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)
        print(f"[保存] {filename} ({len(self.episode_experiences)}ステップ, スコア:{total_score}, 最大タイル:{max_tile})")


# グローバルゲームインスタンス
game: Optional[Game2048Server] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPIのライフサイクル管理"""
    global game
    game = Game2048Server()
    print("=" * 50)
    print("  2048 Experience Collector")
    print("=" * 50)
    print(f"  ブラウザで http://{HOST}:{PORT} を開いてください")
    print(f"  Experienceの保存先: {os.path.abspath(EXPERIENCE_DIR)}/")
    print(f"  終了: Ctrl+C")
    print("=" * 50)
    yield
    # シャットダウン時に未保存のexperienceを保存
    if game and game.episode_experiences:
        game._save_experience()
        print("\n[保存] 未保存のエピソードを保存しました")
    print("\nサーバーを終了しました")


app = FastAPI(lifespan=lifespan)


class StepRequest(BaseModel):
    action: int

# HTML/JS/CSSを埋め込み
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>2048 - Experience Collector</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #faf8ef;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    flex-direction: column;
    overscroll-behavior: none;
    touch-action: pan-x pan-y;
}
.container {
    max-width: 500px;
    width: 100%;
    padding: 12px;
}
h1 {
    font-size: 48px;
    font-weight: bold;
    color: #776e65;
    margin-bottom: 0;
    flex-shrink: 0;
}
.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    gap: 8px;
    flex-wrap: wrap;
}
.scores {
    display: flex;
    gap: 4px;
    flex-shrink: 1;
    min-width: 0;
    flex: 1 1 auto;
}
.score-box {
    background: #bbada0;
    color: white;
    padding: 6px 10px;
    border-radius: 6px;
    text-align: center;
    min-width: 0;
    flex: 1;
    overflow: hidden;
}
.score-box .label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.8;
    white-space: nowrap;
}
.score-box .value {
    font-size: 18px;
    font-weight: bold;
    white-space: nowrap;
}

/* --- Board layout --- */
.board-wrapper {
    position: relative;
    margin-bottom: 20px;
    touch-action: none;
}
.board {
    background: #bbada0;
    border-radius: 8px;
    padding: 12px;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    aspect-ratio: 1;
}
.cell {
    background: #cdc1b4;
    border-radius: 6px;
    position: relative;
    overflow: visible;
}
/* Tile inside a grid cell (static, always correct) */
.tile {
    position: absolute;
    inset: 0;
    border-radius: 6px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-weight: bold;
    font-size: 36px;
    color: #776e65;
    z-index: 2;
}
/* Animation overlay layer */
.anim-layer {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 10;
    overflow: visible;
}
.anim-tile {
    position: absolute;
    border-radius: 6px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-weight: bold;
    font-size: 36px;
    color: #776e65;
    transition: top 120ms ease-in-out, left 120ms ease-in-out;
    will-change: top, left;
}
.anim-tile.no-transition { transition: none !important; }

/* Animations */
.tile-new { animation: pop-in 200ms ease forwards; }
.tile-merged { animation: merge-pop 200ms ease; }
@keyframes pop-in {
    0%   { transform: scale(0); opacity: 0; }
    50%  { transform: scale(1.1); opacity: 1; }
    100% { transform: scale(1); opacity: 1; }
}
@keyframes merge-pop {
    0%   { transform: scale(1); }
    40%  { transform: scale(1.3); }
    100% { transform: scale(1); }
}
.score-add {
    position: fixed;
    animation: score-fly 600ms ease forwards;
    font-size: 18px;
    font-weight: bold;
    color: #776e65;
    pointer-events: none;
    z-index: 100;
}
@keyframes score-fly {
    0%   { opacity: 1; transform: translateY(0); }
    100% { opacity: 0; transform: translateY(-30px); }
}

/* Tile colors */
.tile-2    { background: #eee4da; }
.tile-4    { background: #ede0c8; }
.tile-8    { background: #f2b179; color: white; }
.tile-16   { background: #f59563; color: white; }
.tile-32   { background: #f67c5f; color: white; }
.tile-64   { background: #f65e3b; color: white; }
.tile-128  { background: #edcf72; color: white; font-size: 30px; }
.tile-256  { background: #edcc61; color: white; font-size: 30px; }
.tile-512  { background: #edc850; color: white; font-size: 30px; }
.tile-1024 { background: #edc53f; color: white; font-size: 24px; }
.tile-2048 { background: #edc22e; color: white; font-size: 24px; }
.tile-4096 { background: #3c3a32; color: white; font-size: 24px; }
.tile-8192 { background: #3c3a32; color: white; font-size: 24px; }

.controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.btn {
    background: #8f7a66;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    font-size: 15px;
    font-weight: bold;
    cursor: pointer;
    transition: background 0.2s;
}
.btn:hover { background: #9f8b77; }
.info {
    color: #776e65;
    font-size: 13px;
    text-align: center;
    line-height: 1.6;
}
.info kbd {
    background: #eee;
    border: 1px solid #ccc;
    border-radius: 3px;
    padding: 2px 6px;
    font-size: 11px;
}
.status {
    text-align: center;
    margin-top: 6px;
    font-size: 12px;
    color: #999;
}

/* Hide keyboard hints on touch devices */
@media (hover: none) and (pointer: coarse) {
    .info { display: none; }
}

/* Fine-tune for very small screens */
@media (max-width: 400px) {
    h1 { font-size: 36px; }
    .score-box { min-width: 0; padding: 4px 6px; }
    .score-box .value { font-size: 14px; }
    .container { padding: 8px; }
}
.game-over-overlay {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(238, 228, 218, 0.73);
    z-index: 100;
    justify-content: center;
    align-items: center;
    flex-direction: column;
}
.game-over-overlay.active { display: flex; }
.game-over-msg {
    font-size: 48px;
    font-weight: bold;
    color: #776e65;
    margin-bottom: 20px;
}
.game-over-score {
    font-size: 24px;
    color: #776e65;
    margin-bottom: 30px;
}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>2048</h1>
        <div class="scores">
            <div class="score-box">
                <div class="label">Score</div>
                <div class="value" id="score">0</div>
            </div>
            <div class="score-box">
                <div class="label">Max</div>
                <div class="value" id="max-tile">0</div>
            </div>
            <div class="score-box">
                <div class="label">Steps</div>
                <div class="value" id="steps">0</div>
            </div>
        </div>
    </div>
    <div class="board-wrapper">
        <div class="board" id="board">
            <div class="cell" data-r="0" data-c="0"></div>
            <div class="cell" data-r="0" data-c="1"></div>
            <div class="cell" data-r="0" data-c="2"></div>
            <div class="cell" data-r="0" data-c="3"></div>
            <div class="cell" data-r="1" data-c="0"></div>
            <div class="cell" data-r="1" data-c="1"></div>
            <div class="cell" data-r="1" data-c="2"></div>
            <div class="cell" data-r="1" data-c="3"></div>
            <div class="cell" data-r="2" data-c="0"></div>
            <div class="cell" data-r="2" data-c="1"></div>
            <div class="cell" data-r="2" data-c="2"></div>
            <div class="cell" data-r="2" data-c="3"></div>
            <div class="cell" data-r="3" data-c="0"></div>
            <div class="cell" data-r="3" data-c="1"></div>
            <div class="cell" data-r="3" data-c="2"></div>
            <div class="cell" data-r="3" data-c="3"></div>
        </div>
        <div class="anim-layer" id="anim-layer"></div>
    </div>
    <div class="controls">
        <button class="btn" onclick="resetGame()">New Game</button>
        <div class="info">
            <kbd>&uarr;</kbd> <kbd>&darr;</kbd> <kbd>&larr;</kbd> <kbd>&rarr;</kbd> or <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd>
        </div>
    </div>
    <div class="status" id="status">Experience自動保存: ON</div>
</div>

<div class="game-over-overlay" id="game-over">
    <div class="game-over-msg">Game Over!</div>
    <div class="game-over-score" id="game-over-score"></div>
    <button class="btn" onclick="resetGame()">Try Again</button>
    <div style="margin-top:10px;color:#776e65;font-size:14px;">Experienceを自動保存しました</div>
</div>

<script>
const SLIDE_MS = 120;
let isProcessing = false;
let currentBoard = null;
let prevScore = 0;

const KEY_MAP = {
    'ArrowUp': 0, 'ArrowRight': 1, 'ArrowDown': 2, 'ArrowLeft': 3,
    'w': 0, 'd': 1, 's': 2, 'a': 3,
    'W': 0, 'D': 1, 'S': 2, 'A': 3,
};

const sleep = ms => new Promise(r => setTimeout(r, ms));

// --- Build cell lookup from DOM ---
const cells = {};
document.querySelectorAll('.cell').forEach(el => {
    const r = +el.dataset.r, c = +el.dataset.c;
    if (!cells[r]) cells[r] = {};
    cells[r][c] = el;
});

// Get pixel position of a cell relative to the animation overlay
function getCellRect(r, c) {
    const wrapperRect = document.getElementById('anim-layer').getBoundingClientRect();
    const cellRect = cells[r][c].getBoundingClientRect();
    return {
        top:    cellRect.top  - wrapperRect.top,
        left:   cellRect.left - wrapperRect.left,
        width:  cellRect.width,
        height: cellRect.height,
    };
}

function tileClasses(val) {
    return val > 0 ? ' tile-' + val : '';
}

function tileFontSize(val) {
    if (val >= 1024) return '24px';
    if (val >= 128)  return '30px';
    return '';
}

// ============================================================
// Render board into grid cells (static layer, always correct)
// ============================================================
function renderBoard(board, options) {
    const merged  = (options && options.merged)  || {};
    const spawned = (options && options.spawned) || {};

    for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
            const cell = cells[r][c];
            const val  = board[r][c];

            // Clear any previous tile
            const old = cell.querySelector('.tile');
            if (old) old.remove();

            if (val > 0) {
                const tile = document.createElement('div');
                tile.className = 'tile' + tileClasses(val);
                tile.textContent = val;
                const fs = tileFontSize(val);
                if (fs) tile.style.fontSize = fs;

                const key = r + ',' + c;
                if (spawned[key]) tile.classList.add('tile-new');
                else if (merged[key]) tile.classList.add('tile-merged');

                cell.appendChild(tile);
            }
        }
    }
}

// Create a sliding tile in the animation overlay (pixel positioned)
function createAnimTile(val, r, c) {
    const rect = getCellRect(r, c);
    const el = document.createElement('div');
    el.className = 'anim-tile no-transition' + tileClasses(val);
    el.textContent = val > 0 ? val : '';
    const fs = tileFontSize(val);
    if (fs) el.style.fontSize = fs;
    el.style.top    = rect.top    + 'px';
    el.style.left   = rect.left   + 'px';
    el.style.width  = rect.width  + 'px';
    el.style.height = rect.height + 'px';
    return el;
}

// ============================================================
// Client-side move simulation (for slide animation only)
// ============================================================
function simulateMove(board, action) {
    const size = 4;
    const tiles = [];
    const newBoard = Array.from({length: size}, () => Array(size).fill(0));

    for (let i = 0; i < size; i++) {
        const line = [], coords = [];
        for (let j = 0; j < size; j++) {
            let r, c;
            switch (action) {
                case 0: r = j; c = i; break;           // UP
                case 1: r = i; c = size - 1 - j; break; // RIGHT
                case 2: r = size - 1 - j; c = i; break; // DOWN
                case 3: r = i; c = j; break;           // LEFT
            }
            line.push(board[r][c]);
            coords.push({ r, c });
        }

        const nonZero = [];
        for (let j = 0; j < size; j++) {
            if (line[j] !== 0) nonZero.push({ idx: j, val: line[j] });
        }

        let pos = 0, k = 0;
        while (k < nonZero.length) {
            let dr, dc;
            switch (action) {
                case 0: dr = pos; dc = i; break;
                case 1: dr = i; dc = size - 1 - pos; break;
                case 2: dr = size - 1 - pos; dc = i; break;
                case 3: dr = i; dc = pos; break;
            }
            if (k + 1 < nonZero.length && nonZero[k].val === nonZero[k + 1].val) {
                // Merge
                tiles.push({ fR: coords[nonZero[k].idx].r,     fC: coords[nonZero[k].idx].c,
                              tR: dr, tC: dc, val: nonZero[k].val, merged: true });
                tiles.push({ fR: coords[nonZero[k + 1].idx].r, fC: coords[nonZero[k + 1].idx].c,
                              tR: dr, tC: dc, val: nonZero[k + 1].val, absorbed: true });
                newBoard[dr][dc] = nonZero[k].val * 2;
                k += 2;
            } else {
                tiles.push({ fR: coords[nonZero[k].idx].r, fC: coords[nonZero[k].idx].c,
                              tR: dr, tC: dc, val: nonZero[k].val });
                newBoard[dr][dc] = nonZero[k].val;
                k += 1;
            }
            pos++;
        }
    }
    return { tiles, newBoard };
}

// ============================================================
// Score fly-up effect
// ============================================================
function showScoreAdd(delta) {
    if (delta <= 0) return;
    const el = document.getElementById('score');
    const rect = el.getBoundingClientRect();
    const fly = document.createElement('div');
    fly.className = 'score-add';
    fly.textContent = '+' + delta;
    fly.style.left = rect.left + 'px';
    fly.style.top  = rect.top  + 'px';
    document.body.appendChild(fly);
    setTimeout(() => fly.remove(), 650);
}

function fitScoreValue(el) {
    const text = el.textContent;
    if (text.length >= 6) el.style.fontSize = '12px';
    else if (text.length >= 5) el.style.fontSize = '14px';
    else el.style.fontSize = '';
}

function updateScores(state) {
    const delta = state.score - prevScore;
    const scoreEl   = document.getElementById('score');
    const maxEl     = document.getElementById('max-tile');
    const stepsEl   = document.getElementById('steps');
    scoreEl.textContent = state.score;
    maxEl.textContent   = state.max_tile;
    stepsEl.textContent = state.steps;
    fitScoreValue(scoreEl);
    fitScoreValue(maxEl);
    fitScoreValue(stepsEl);
    showScoreAdd(delta);
    prevScore = state.score;
}

// ============================================================
// API helper
// ============================================================
async function apiCall(endpoint, data) {
    const opts = { method: 'POST', headers: { 'Content-Type': 'application/json' } };
    if (data) opts.body = JSON.stringify(data);
    const res = await fetch(endpoint, opts);
    return res.json();
}

// ============================================================
// Game actions
// ============================================================
async function resetGame() {
    document.getElementById('game-over').classList.remove('active');
    document.getElementById('anim-layer').innerHTML = '';
    const state = await apiCall('/api/reset');
    currentBoard = state.board;
    prevScore = state.score;

    const spawned = {};
    for (let r = 0; r < 4; r++)
        for (let c = 0; c < 4; c++)
            if (state.board[r][c] > 0) spawned[r + ',' + c] = true;

    renderBoard(state.board, { spawned });
    updateScores(state);
    document.getElementById('status').textContent = 'Experience自動保存: ON';
}

async function doStep(action) {
    if (isProcessing) return;
    isProcessing = true;
    try {
        const oldBoard = currentBoard;

        // Simulate move locally first (instant)
        const sim = simulateMove(oldBoard, action);

        // Quick check: if nothing moved, skip
        const moved = sim.tiles.some(t => t.fR !== t.tR || t.fC !== t.tC);
        if (!moved && !sim.tiles.some(t => t.merged)) {
            isProcessing = false;
            return;
        }

        // Start API call AND slide animation in parallel
        const apiPromise = apiCall('/api/step', { action });

        const animLayer = document.getElementById('anim-layer');

        // --- Phase 1: hide grid tiles, create slide overlay ---
        document.querySelectorAll('.cell .tile').forEach(t => { t.style.visibility = 'hidden'; });

        animLayer.innerHTML = '';
        const animEls = [];
        for (const t of sim.tiles) {
            const el = createAnimTile(t.val, t.fR, t.fC);
            animLayer.appendChild(el);
            animEls.push({ el, t });
        }
        void animLayer.offsetHeight; // force reflow

        // --- Phase 2: slide tiles to new positions ---
        for (const { el, t } of animEls) {
            el.classList.remove('no-transition');
            const rect = getCellRect(t.tR, t.tC);
            el.style.top  = rect.top  + 'px';
            el.style.left = rect.left + 'px';
        }

        // Wait for both animation and API to finish
        const [state] = await Promise.all([apiPromise, sleep(SLIDE_MS + 10)]);

        if (!state.is_legal) {
            // Restore board if server says illegal
            renderBoard(oldBoard, {});
            animLayer.innerHTML = '';
            isProcessing = false;
            return;
        }

        // --- Phase 3: render final board in grid (always correct) ---
        const finalBoard = state.board;

        const merged  = {};
        const spawned = {};
        for (const t of sim.tiles) {
            if (t.merged) merged[t.tR + ',' + t.tC] = true;
        }
        for (let r = 0; r < 4; r++)
            for (let c = 0; c < 4; c++)
                if (sim.newBoard[r][c] === 0 && finalBoard[r][c] > 0)
                    spawned[r + ',' + c] = true;

        renderBoard(finalBoard, { merged, spawned });
        animLayer.innerHTML = '';

        currentBoard = finalBoard;
        updateScores(state);

        if (state.terminated) {
            await sleep(200);
            document.getElementById('game-over-score').textContent =
                'Score: ' + state.score + ' | Max Tile: ' + state.max_tile;
            document.getElementById('game-over').classList.add('active');
        }
    } finally {
        isProcessing = false;
    }
}

// ============================================================
// Input handlers
// ============================================================
document.addEventListener('keydown', (e) => {
    if (e.key in KEY_MAP) {
        e.preventDefault();
        doStep(KEY_MAP[e.key]);
    }
});

let touchStartX = 0, touchStartY = 0;
const boardWrapper = document.querySelector('.board-wrapper');
boardWrapper.addEventListener('touchstart', (e) => {
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY;
}, { passive: true });
boardWrapper.addEventListener('touchmove', (e) => {
    e.preventDefault();
}, { passive: false });
boardWrapper.addEventListener('touchend', (e) => {
    const dx = e.changedTouches[0].clientX - touchStartX;
    const dy = e.changedTouches[0].clientY - touchStartY;
    const absDx = Math.abs(dx), absDy = Math.abs(dy);
    if (Math.max(absDx, absDy) < 30) return;
    e.preventDefault();
    if (absDx > absDy) doStep(dx > 0 ? 1 : 3);
    else doStep(dy > 0 ? 2 : 0);
});

// Start
resetGame();

// Save experience when tab is closed or hidden
function saveBeforeLeave() {
    navigator.sendBeacon('/api/save');
}
window.addEventListener('beforeunload', saveBeforeLeave);
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') saveBeforeLeave();
});
</script>
</body>
</html>"""


# ============================================================
# FastAPI routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/api/reset")
async def api_reset():
    assert game is not None
    state = game.reset()
    return JSONResponse(content=state)


@app.post("/api/step")
async def api_step(req: StepRequest):
    assert game is not None
    state = game.step(req.action)
    return JSONResponse(content=state)


@app.post("/api/save")
async def api_save():
    assert game is not None
    if game.episode_experiences:
        game._save_experience()
        game.episode_experiences = []
    return JSONResponse(content={"ok": True})


def main():
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
