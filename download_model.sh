#!/bin/bash
# 从 hf-mirror.com 直接下载 Qwen2.5-7B-Instruct，绕过 XetHub
set -e

MODEL_DIR="models/Qwen2.5-7B-Instruct"
BASE_URL="https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct/resolve/main"

mkdir -p "$MODEL_DIR"

FILES=(
  config.json
  generation_config.json
  model.safetensors.index.json
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
)

download_file() {
  # 兼容不同 wget 版本：部分环境（如 BusyBox）不支持 --show-progress
  # 优先 wget，其次 curl；都没有则报错退出
  local out="$1"
  local url="$2"

  if command -v wget >/dev/null 2>&1; then
    if wget --help 2>&1 | grep -q -- '--show-progress'; then
      wget -c --show-progress -O "$out" "$url"
    else
      # 老 wget 通常支持 --progress=bar:force:noscroll；再不行就无进度降级
      if wget --help 2>&1 | grep -q -- '--progress'; then
        wget -c --progress=bar:force:noscroll -O "$out" "$url" || wget -c -O "$out" "$url"
      else
        wget -c -O "$out" "$url"
      fi
    fi
    return 0
  fi

  if command -v curl >/dev/null 2>&1; then
    # -C - 断点续传；-L 跟随跳转
    curl -L -C - -o "$out" "$url"
    return 0
  fi

  echo "错误：未找到 wget 或 curl，无法下载 $url" >&2
  return 127
}

for f in "${FILES[@]}"; do
  if [ -f "$MODEL_DIR/$f" ]; then
    echo "[跳过] $f 已存在"
  else
    echo "[下载] $f ..."
    download_file "$MODEL_DIR/$f" "$BASE_URL/$f"
  fi
done

# 有些仓库不提供该文件（404 属于正常），下载不到就跳过
OPTIONAL_FILES=(
  special_tokens_map.json
  LICENSE
  README.md
)

for f in "${OPTIONAL_FILES[@]}"; do
  if [ -f "$MODEL_DIR/$f" ]; then
    echo "[跳过] (optional) $f 已存在"
  else
    echo "[下载] (optional) $f ..."
    if ! download_file "$MODEL_DIR/$f" "$BASE_URL/$f"; then
      echo "[跳过] (optional) $f 下载失败（可能是 404），不影响使用。"
      rm -f "$MODEL_DIR/$f"
    fi
  fi
done

echo ""
echo "下载完成！模型保存在: $MODEL_DIR"
echo "使用方式: --model_name $MODEL_DIR"