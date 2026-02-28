"""
ブラウザUIのAI自動プレイを録画するスクリプト

必要パッケージ:
    uv add playwright
    uv run playwright install chromium

使い方:
    uv run python record_video.py
"""

import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path


async def wait_for_server(url: str, timeout: int = 15):
    """サーバーが起動するまで待つ"""
    import urllib.request
    import urllib.error

    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except (urllib.error.URLError, ConnectionError, OSError):
            await asyncio.sleep(0.5)
    return False


async def record():
    output_dir = Path("./videos")
    output_dir.mkdir(exist_ok=True)

    # 1. サーバー起動
    print("[録画] サーバーを起動中...")
    server = subprocess.Popen(
        [sys.executable, "play_browser.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        if not await wait_for_server("http://localhost:8080"):
            print("[エラー] サーバーの起動がタイムアウトしました")
            return

        print("[録画] サーバー起動完了")

        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            context = await browser.new_context(
                viewport={"width": 540, "height": 720},
                record_video_dir=str(output_dir),
                record_video_size={"width": 540, "height": 720},
            )
            page = await context.new_page()
            await page.goto("http://localhost:8080")
            await page.wait_for_timeout(1000)

            print("[録画] AI自動プレイを開始...")

            # コンソールログを表示（デバッグ用）
            page.on("console", lambda msg: print(f"  [browser] {msg.text}") if msg.type == "error" else None)

            # AI Playボタンをクリック
            await page.click("#auto-play-btn")

            # Game Overオーバーレイが表示されるまでポーリングで待つ (最大5分)
            game_over = False
            last_score = 0
            last_max = 0
            last_steps = 0
            for i in range(3000):  # 最大300秒 (100ms * 3000)
                is_active = await page.evaluate(
                    "document.getElementById('game-over').classList.contains('active')"
                )
                if is_active:
                    game_over = True
                    break
                # 2秒ごとに進行状況を表示
                if i % 20 == 0 and i > 0:
                    status = await page.evaluate("""(() => {
                        return {
                            score: document.getElementById('score').textContent,
                            max: document.getElementById('max-tile').textContent,
                            steps: document.getElementById('steps').textContent,
                        };
                    })()""")
                    score = status.get("score", "?")
                    max_tile = status.get("max", "?")
                    steps = status.get("steps", "?")
                    if score != last_score or max_tile != last_max or steps != last_steps:
                        print(f"  [進行中] Score: {score} | Max Tile: {max_tile} | Steps: {steps}")
                        last_score = score
                        last_max = max_tile
                        last_steps = steps
                await page.wait_for_timeout(100)

            if game_over:
                print("[録画] ゲーム終了を検出")
            else:
                print("[警告] タイムアウト: ゲーム終了を検出できませんでした")

            # 少し待ってから終了（Game Overの表示を撮る）
            await page.wait_for_timeout(3000)

            # コンテキストを閉じると動画が保存される
            await context.close()
            await browser.close()

        # 保存された動画ファイルを探してMP4に変換
        video_files = list(output_dir.glob("*.webm"))
        if video_files:
            latest_webm = max(video_files, key=lambda f: f.stat().st_mtime)
            mp4_path = latest_webm.with_suffix(".mp4")

            if shutil.which("ffmpeg"):
                print(f"[変換] {latest_webm.name} → {mp4_path.name}")
                result = subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(latest_webm),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        str(mp4_path),
                    ],
                    capture_output=True,
                )
                if result.returncode == 0:
                    latest_webm.unlink()  # webmを削除
                    print(f"[完了] 録画保存先: {mp4_path}")
                else:
                    print(f"[警告] ffmpeg変換失敗。webmのまま保存: {latest_webm}")
                    print(result.stderr.decode())
            else:
                print(f"[完了] 録画保存先: {latest_webm}")
                print(f"[ヒント] ffmpegをインストールすれば自動でMP4に変換されます")
        else:
            print("[完了] 録画ファイルが見つかりません。./videos/ を確認してください")

    finally:
        server.terminate()
        server.wait()
        print("[録画] サーバーを停止しました")


if __name__ == "__main__":
    asyncio.run(record())
