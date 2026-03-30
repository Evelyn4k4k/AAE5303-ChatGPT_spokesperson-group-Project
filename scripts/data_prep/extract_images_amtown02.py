from pathlib import Path
from rosbags.highlevel import AnyReader
import cv2
import numpy as np

bag_path = Path("data/raw_bags/AMtown02.bag")
out_dir = Path("data/extracted_images/images")
timestamp_file = Path("data/extracted_images/timestamps.txt")

image_topic = "/left_camera/image/compressed"

out_dir.mkdir(parents=True, exist_ok=True)
timestamp_file.parent.mkdir(parents=True, exist_ok=True)

def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

count = 0

with AnyReader([bag_path]) as reader, open(timestamp_file, "w", encoding="utf-8") as tf:
    connections = [c for c in reader.connections if c.topic == image_topic]

    print(f"Reading bag: {bag_path}")
    print(f"Extracting topic: {image_topic}")

    for conn, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, conn.msgtype)

        t = stamp_to_sec(msg.header.stamp)

        # CompressedImage.data -> numpy bytes
        img_bytes = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Warning: failed to decode image at t={t:.9f}")
            continue

        # 文件名用时间戳，保留9位小数，便于后续对齐
        fname = f"{t:.9f}.png"
        fpath = out_dir / fname

        cv2.imwrite(str(fpath), image)
        tf.write(f"{t:.9f} images/{fname}\n")

        count += 1
        if count % 1000 == 0:
            print(f"Saved {count} images...")

print(f"Done. Total images saved: {count}")
print(f"Images directory: {out_dir}")
print(f"Timestamps file: {timestamp_file}")