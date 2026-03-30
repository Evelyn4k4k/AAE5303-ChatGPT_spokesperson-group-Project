from pathlib import Path
from rosbags.highlevel import AnyReader

bag_path = Path("data/raw_bags/AMtown02.bag")

with AnyReader([bag_path]) as reader:
    print("Topics in bag:")
    for conn in reader.connections:
        print(f"{conn.topic} | {conn.msgtype}")
