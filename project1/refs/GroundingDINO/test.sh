#!/bin/bash
# ==============================================
# 批量运行 GroundingDINO + SAM 联动分割
# 作者：WOODENMAN
# 处理对象：car1.jpg ~ car25.jpg
# ==============================================

# === 路径设置 ===
ROOT="/home/cxx/HWs/CS290U/project1/refs/GroundingDINO"
SEG_PATH="../segment-anything"
CONFIG="$ROOT/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHT="$ROOT/models/groundingdino_swint_ogc.pth"
SAM_WEIGHT="$SEG_PATH/models/sam_vit_h_4b8939.pth"
IMG_DIR="$SEG_PATH/data"
OUT_DIR="$ROOT/outputs/cars"

# === 检查依赖 ===
if [ ! -f "$DINO_WEIGHT" ]; then
  echo "❌ Missing DINO weight file: $DINO_WEIGHT"
  exit 1
fi

if [ ! -f "$SAM_WEIGHT" ]; then
  echo "❌ Missing SAM weight file: $SAM_WEIGHT"
  exit 1
fi

mkdir -p "$OUT_DIR"

# === 批量处理 car1 ~ car25 ===
for i in $(seq 1 25); do
  IMG="$IMG_DIR/car${i}.jpg"
  if [ ! -f "$IMG" ]; then
    echo "⚠️ 跳过: 未找到 $IMG"
    continue
  fi

  echo "==============================="
  echo "🚗 处理 car${i}.jpg ..."
  echo "==============================="

  python "$ROOT/combine.py" \
    --dino-config "$CONFIG" \
    --dino-weights "$DINO_WEIGHT" \
    --sam-weights "$SAM_WEIGHT" \
    --image "$IMG" \
    --text "car, vehicle, automobile, wheel, window" \
    --output "$OUT_DIR" \
    --box-thresh 0.2 \
    --text-thresh 0.2 \
    --multimask 1

  echo "✅ 完成 car${i}.jpg"
done

echo "🎉 所有图片已完成！输出目录: $OUT_DIR"
