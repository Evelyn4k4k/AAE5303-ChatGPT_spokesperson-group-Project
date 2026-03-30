from pathlib import Path
import cv2

src_dir = Path("data/extracted_images/images")
src_ts = Path("data/extracted_images/timestamps.txt")

dst_dir = Path("data/extracted_images_2x/images")
dst_ts = Path("data/extracted_images_2x/timestamps.txt")

scale = 0.5  # 2x downsample

dst_dir.mkdir(parents=True, exist_ok=True)
dst_ts.parent.mkdir(parents=True, exist_ok=True)

count = 0

with open(src_ts, "r", encoding="utf-8") as fin, open(dst_ts, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        t, rel_path = line.split()
        src_img_path = Path("data/extracted_images") / rel_path
        img = cv2.imread(str(src_img_path))

        if img is None:
            print(f"Failed to read: {src_img_path}")
            continue

        h, w = img.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)

        img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        dst_img_path = dst_dir / src_img_path.name
        cv2.imwrite(str(dst_img_path), img_small)

        fout.write(f"{t} images/{src_img_path.name}\n")

        count += 1
        if count % 1000 == 0:
            print(f"Saved {count} downsampled images...")

print(f"Done. Total saved: {count}")
print(f"Output dir: {dst_dir}")
print(f"Timestamps: {dst_ts}")