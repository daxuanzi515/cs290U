#!/bin/bash
# 批量运行 SAM 分割 + mask 可视化
# 适用于 car1.jpg ~ car25.jpg

# ======= 路径配置 =======
ROOT="/home/cxx/HWs/CS290U/project1/refs/segment-anything"
CHECKPOINT="$ROOT/models/sam_vit_h_4b8939.pth"
INPUT_DIR="$ROOT/data"
OUTPUT_DIR="$ROOT/outputs/cars"
AMG_SCRIPT="$ROOT/scripts/amg.py"
VIS_SCRIPT="$ROOT/scripts/mask_add.py"

# ======= 检查模型文件 =======
if [ ! -f "$CHECKPOINT" ]; then
  echo "❌ SAM checkpoint not found: $CHECKPOINT"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ======= 批量处理 car1–car25 =======
for i in $(seq 1 25); do
  IMG="$INPUT_DIR/car${i}.jpg"
  JSON="$OUTPUT_DIR/car${i}.json"
  OVERLAY="$OUTPUT_DIR/car${i}_overlay.jpg"

  if [ ! -f "$IMG" ]; then
    echo "⚠️ 跳过：未找到图片 $IMG"
    continue
  fi

  echo "==============================="
  echo "🚗 处理图片 car${i}.jpg"
  echo "==============================="

  # 1️⃣ 生成 mask JSON
  python "$AMG_SCRIPT" \
    --checkpoint "$CHECKPOINT" \
    --model-type vit_h \
    --input "$IMG" \
    --output "$OUTPUT_DIR" \
    --convert-to-rle

  # 2️⃣ 叠加可视化
  python "$VIS_SCRIPT" \
    --image "$IMG" \
    --json "$JSON" \
    --output "$OVERLAY"

  echo "✅ car${i} 完成"
done

echo "🎉 所有图片已处理完成，结果保存在：$OUTPUT_DIR"
