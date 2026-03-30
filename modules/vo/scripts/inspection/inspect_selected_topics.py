from pathlib import Path
from rosbags.highlevel import AnyReader

bag_path = Path("data/raw_bags/AMtown02.bag")
targets = {
    "/left_camera/image/compressed",
    "/dji_osdk_ros/local_position",
    "/dji_osdk_ros/attitude",
    "/dji_osdk_ros/rtk_position",
    "/dji_osdk_ros/vo_position",
}

with AnyReader([bag_path]) as reader:
    connections = [c for c in reader.connections if c.topic in targets]
    shown = set()
    for conn, timestamp, rawdata in reader.messages(connections=connections):
        if conn.topic in shown:
            continue
        msg = reader.deserialize(rawdata, conn.msgtype)
        print("\nTOPIC:", conn.topic)
        print("TYPE :", conn.msgtype)
        print("MSG  :", msg)
        shown.add(conn.topic)
        if shown == targets:
            break