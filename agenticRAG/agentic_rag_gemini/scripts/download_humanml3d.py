import json
import re
import sys
import hashlib
from pathlib import Path

# Cấu hình đường dẫn
SCRIPT_DIR = Path(__file__).resolve().parent
SERVICE_ROOT = SCRIPT_DIR.parent
OUTPUT_PATH = SERVICE_ROOT / "data" / "knowledge_base" / "humanml3d_descriptions.jsonl"
DATASET_NAME = "TeoGchx/HumanML3D"

def clean_text(raw: str) -> list[str]:
    """Loại bỏ POS tags sau dấu '#' và dọn dẹp text."""
    if not raw: return []
    descriptions = []
    # Xử lý nếu raw là list hoặc string
    lines = raw if isinstance(raw, list) else str(raw).split("\n")
    
    for line in lines:
        line = line.strip()
        if not line: continue
        # Chỉ lấy phần trước dấu '#' (nơi POS tags bắt đầu)
        text = line.split("#")[0].strip()
        if text and len(text) > 3:
            text = re.sub(r"\s+", " ", text)
            descriptions.append(text)
    return descriptions

def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed. Run: pip install datasets")
        return 1

    print(f"🚀 Khởi động: Tải {DATASET_NAME}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Sử dụng streaming=True để không ngốn RAM
    # Hoặc load từng split một thay vì dồn tất cả vào list
    try:
        ds = load_dataset(DATASET_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Lỗi tải dataset: {e}")
        return 1

    seen_hashes = set() # Sử dụng Hash thay vì string dài để tiết kiệm RAM
    count = 0

    print("🧹 Đang dọn dẹp POS tags và ghi file (Sử dụng Stream)...")
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for split_name in ds:
            print(f"📦 Đang xử lý split: {split_name}...")
            # Lặp trực tiếp qua split, không dùng .extend()
            for row in ds[split_name]:
                raw_text = row.get("caption") or row.get("text") or ""
                clean_descs = clean_text(raw_text)
                
                if not clean_descs: continue
                primary = clean_descs[0]

                # Deduplicate bằng Hash (Tiết kiệm bộ nhớ cực lớn)
                h = hashlib.md5(primary.lower().encode()).hexdigest()
                if h in seen_hashes: continue
                seen_hashes.add(h)

                meta = row.get("meta_data") or {}
                entry = {
                    "id": str(row.get("id", row.get("name", count))),
                    "text_description": primary,
                    "motion_prompt": primary,
                    "alt_descriptions": clean_descs[1:] if len(clean_descs) > 1 else [],
                    "duration": meta.get("duration") if isinstance(meta, dict) else None,
                    "num_frames": meta.get("num_frames") if isinstance(meta, dict) else None,
                }

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
                
                if count % 2000 == 0:
                    print(f" ✅ Đã xử lý {count} dòng...")

    print(f"\n🎉 Thành công! Đã lưu {count} entry sạch sẽ vào {OUTPUT_PATH}")
    return 0

if __name__ == "__main__":
    sys.exit(main())