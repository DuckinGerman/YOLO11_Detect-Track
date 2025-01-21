import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

def initialize_yolo(model_path="/Users/tianqili/Desktop/yolo11+deepsort/exp2/weights/best.pt"):
    return YOLO(model_path)

def initialize_tracker():
    return DeepSort(
        max_age=30,            # 最大未更新帧数
        n_init=4,              # 初始化需要的最少帧数
        max_iou_distance=0.8,  # 最大IOU匹配距离
        embedder="mobilenet",  # 使用 mobilenet 作为特征提取器
        half=True              # 使用半精度 (FP16) 提升性能（需要支持）
    )

def initialize_video(video_path, output_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cant open file：{video_path}")
        exit(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"frame resolution: {width}x{height}, FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    return cap, out, width, height

def process_frame(frame, model, tracker, class_names):
    """for single frame , detect and track"""
    results = model(frame, conf=0.3, iou=0.4) # get result from detector
    detections = []

    for result in results:
        boxes = result.boxes.xywh  # box info
        confidences = result.boxes.conf  # conf
        class_ids = result.boxes.cls  # cls

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            detections.append((box, conf, int(class_id)))

    # for every frame deepsort
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    for track in tracked_objects:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        sort_x1, sort_y1, sort_x2, sort_y2 = map(int, track.to_ltrb())
        class_id = track.det_class
        class_name = class_names[class_id]

        w = (sort_x2 - sort_x1)
        h = (sort_y2 - sort_y1)
        re_x1 = int(sort_x1 - w/2)  # 左上角坐标
        re_x2 = int(sort_x1 + w/2)
        re_y1 = int(sort_y1 + h/2)
        re_y2 = int(sort_y1 - h/2)

        cv2.rectangle(frame, (re_x1, re_y1), (re_x2, re_y2), (0, 255, 0), 2)
        label = f"ID:{track_id} {class_name}"
        cv2.putText(frame, label, (re_x1, re_y1 - h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return frame

def main():
    # video_path = 0
    video_path = "/Users/tianqili/Downloads/video_test/MOT16-07-raw.mp4"
    output_path = "/Users/tianqili/Downloads/camera_deep_07.mp4"

    model = initialize_yolo()
    tracker = initialize_tracker()

    cap, out, width, height = initialize_video(video_path, output_path)

    all_frame_start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # 检查是否到达视频结尾
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Reading video is over ")
                break
            else:
                print("Reading current frame wrong !")
                continue

        frame_count += 1
        single_frame_start_time = time.time() # 单帧开始时间

        # process
        class_names = list(model.names.values())
        processed_frame = process_frame(frame, model, tracker, class_names)
        single_frame_end_time = time.time() # 单帧结束时间

        # show result
        cv2.imshow("YOLO11 + DeepSORT", processed_frame)

        # write video
        out.write(processed_frame)
        single_frame_processing_time = single_frame_end_time - single_frame_start_time
        print(f"Single frame time: {single_frame_processing_time:.2f} s")

        # exit with q
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
