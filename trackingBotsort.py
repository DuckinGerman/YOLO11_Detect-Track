from collections import defaultdict
import cv2
from ultralytics import YOLO
import time

def initialize_yolo(model_path="/Users/tianqili/Desktop/yolo11+deepsort/4cls_200/weights/best.pt"):
    """初始化 YOLO 检测器"""
    return YOLO(model_path)

def initialize_tracker():
    """初始化追踪器"""
    track_history = defaultdict(list)
    tracker_name = "botsort.yaml"
    return  tracker_name, track_history

def initialize_video(video_path, output_path):
    """打开视频文件或摄像头，初始化输出视频"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频：{video_path}")
        exit(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"视频分辨率: {width}x{height}, 帧率: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    return cap, out, width, height

def process_frame(frame, model, tracker_name):
    """处理单帧，进行目标检测和跟踪"""
    results = list(model.track(frame, tracker_name,conf=0.3, iou=0.4,persist=True,save=False)) # get result
    # detections = []
    if len(results) == 0 or frame is None or frame.size == 0:
        print("警告：检测结果为空，跳过当前帧")
        return frame

    # for result in results:
    #     boxes = result.boxes.xywh.cpu()  # box info
    #     confidences = result.boxes.conf.cpu()  # conf
    #     class_ids = result.boxes.cls.cpu()  # cls

    # 使用 Ultralytics 内置绘制 result （def plot） 中修改
    frame = results[0].plot()

    # if frame is None or frame.size == 0:
        # print(f"警告：annotated_frame 无效，跳过帧 {frame_count}")
        # continue
        # 绘制自定义轨迹
        # for box, track_id in zip(boxes, track_ids):
        #     cx, cy, w, h = box
        #     track_history[track_id].append((float(cx), float(cy)))

        # 限制轨迹点数量
        # if len(track_history[track_id]) > 30:
        # track_history[track_id].pop(0)

        # 绘制轨迹线
        # points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
        # cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

    return frame

def main():
    # video_path = 0
    video_path = "/Users/tianqili/Downloads/video_test/MOT16-07-raw.mp4"
    output_path = "/Users/tianqili/Downloads/MOT16-07-raw-bot200.mp4"

    model = initialize_yolo()
    tracker_name, track_history = initialize_tracker()

    cap, out, width, height = initialize_video(video_path, output_path)

    all_frame_start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # 检查是否到达视频结尾
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                print("视频读取结束。")
                break
            else:
                print("读取帧时出现错误，跳过当前帧。")
                continue

        frame_count += 1
        single_frame_start_time = time.time() # 单帧开始时间

        # 处理单帧
        processed_frame = process_frame(frame, model, tracker_name)
        single_frame_end_time = time.time() # 单帧结束时间

        # 显示结果
        cv2.imshow("YOLO11 Tracking", processed_frame)

        # 写入输出视频
        out.write(processed_frame)
        single_frame_processing_time = single_frame_end_time - single_frame_start_time
        print(f"Single frame time: {single_frame_processing_time:.2f} s")

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    all_frame_end_time = time.time()
    all_frame_processing_time = all_frame_end_time - all_frame_start_time
    print(f"Total {frame_count} frames。Saved to : {output_path}")
    print(f"All done for time: {all_frame_processing_time:.2f} s")

if __name__ == "__main__":
    main()
