---
description: WSL2環境でAIブラウザ（browser_subagent）を使うためのXvfb+Chrome CDP起動手順
---

WSL2環境で `browser_subagent` を使う前に、以下を実行してください。

## 前提
- `npx playwright install-deps chromium` と `npx playwright install chromium` が完了済み
- `xvfb` がインストール済み（`sudo apt-get install -y xvfb`）

## 手順

// turbo-all

1. Xvfbが起動しているか確認する。起動していなければ起動する
```bash
ps aux | grep '[X]vfb' || Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
```

2. 9222ポートが空いているか確認する。別プロセスに掴まれていたら解放する
```bash
ss -ltnp | grep 9222 && echo "WARNING: 9222が使用中。pkill socat 等で解放してください" || echo "OK: 9222は空き"
```

3. PlaywrightのChromiumをCDPモード（9222番）で起動する
```bash
CHROME_BIN=$(find ~/.cache/ms-playwright -name "chrome" -type f | head -1)
DISPLAY=:99 "$CHROME_BIN" \
  --remote-debugging-port=9222 \
  --remote-debugging-address=127.0.0.1 \
  --no-first-run --no-default-browser-check --no-sandbox --disable-gpu \
  --user-data-dir="$HOME/.gemini/antigravity-browser-profile" &
```

4. CDPの疎通を確認する（JSONが返ればOK）
```bash
sleep 2 && curl -s http://127.0.0.1:9222/json/version | head -3
```

5. 上記が成功したら `browser_subagent` を使ってブラウザ操作を行う

## トラブルシューティング

- **CDPが応答しない**: `pkill -f antigravity-browser-profile` で既存Chromeを停止してから手順3を再実行
- **Xvfbが起動しない**: `/tmp/.X11-unix` のパーミッション問題の場合あり。`Xvfb :99` を別のディスプレイ番号（`:98` 等）で試す
- **Chromiumが見つからない**: `npx -y playwright install chromium` を再実行
