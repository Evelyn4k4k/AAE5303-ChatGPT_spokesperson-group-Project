from pathlib import Path
from bisect import bisect_left
from rosbags.highlevel import AnyReader

bag_path = Path("data/raw_bags/AMtown02.bag")
timestamps_path = Path("data/extracted_images/timestamps.txt")
out_path = Path("data/ground_truth/ground_truth_synced.txt")

pos_topic = "/dji_osdk_ros/local_position"
att_topic = "/dji_osdk_ros/attitude"

# 允许的最大时间差（秒）
max_diff = 0.05  # 50 ms，先用这个；不够再调到 0.1

def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

# 读图像时间戳
image_times = []
with open(timestamps_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        t = float(line.split()[0])
        image_times.append(t)

print(f"Loaded image timestamps: {len(image_times)}")

positions = []
attitudes = []

# 从 bag 读位置和姿态
with AnyReader([bag_path]) as reader:
    connections = [c for c in reader.connections if c.topic in {pos_topic, att_topic}]
    count = 0
    for conn, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, conn.msgtype)
        t = stamp_to_sec(msg.header.stamp)

        if conn.topic == pos_topic:
            positions.append((t, msg.point.x, msg.point.y, msg.point.z))
        elif conn.topic == att_topic:
            q = msg.quaternion
            attitudes.append((t, q.x, q.y, q.z, q.w))

        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} GT messages...")

print(f"Loaded positions: {len(positions)}")
print(f"Loaded attitudes: {len(attitudes)}")

# 为二分查找准备时间轴
pos_times = [p[0] for p in positions]
att_times = [a[0] for a in attitudes]

def nearest_sample(samples, times, t_query, max_diff):
    idx = bisect_left(times, t_query)
    candidates = []
    if idx < len(samples):
        candidates.append(samples[idx])
    if idx > 0:
        candidates.append(samples[idx - 1])

    if not candidates:
        return None

    best = min(candidates, key=lambda s: abs(s[0] - t_query))
    if abs(best[0] - t_query) <= max_diff:
        return best
    return None

matched = []
miss_pos = 0
miss_att = 0

for t_img in image_times:
    p = nearest_sample(positions, pos_times, t_img, max_diff)
    a = nearest_sample(attitudes, att_times, t_img, max_diff)

    if p is None:
        miss_pos += 1
        continue
    if a is None:
        miss_att += 1
        continue

    _, x, y, z = p
    _, qx, qy, qz, qw = a
    matched.append((t_img, x, y, z, qx, qy, qz, qw))

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    for row in matched:
        f.write(
            f"{row[0]:.9f} {row[1]:.9f} {row[2]:.9f} {row[3]:.9f} "
            f"{row[4]:.9f} {row[5]:.9f} {row[6]:.9f} {row[7]:.9f}\n"
        )

print(f"Saved synced ground truth to: {out_path}")
print(f"Matched image poses: {len(matched)} / {len(image_times)}")
print(f"Missed position matches: {miss_pos}")
print(f"Missed attitude matches: {miss_att}")