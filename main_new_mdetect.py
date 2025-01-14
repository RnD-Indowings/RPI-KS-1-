import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import serial 
import time
from mavsdk import System
from mavsdk.offboard import OffboardError
import connect
import torch
import cv2
import numpy as np
import math
import connect
import asyncio
import random
import threading
import os
import subprocess
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from pymavlink import mavutil
from mavsdk import System
from models.common import DetectMultiBackend

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


CHANNEL_15_THRESHOLD = 1500
connection = mavutil.mavlink_connection('udp:127.0.0.1:14550')
terminal_opened = False
lock = threading.Lock()


def run_program_after_delay(center_x, center_y):
    global terminal_opened
    with lock:
        if terminal_opened:
            return
        terminal_opened = True
        time.sleep(0)
    
    # Construct the command to open the terminal and run the script
    command = f'gnome-terminal -- bash -c "python3 localization.py {center_x} {center_y}; exec bash"'
    
    # Execute the command to open the terminal and run the script
    os.system(command)

def listen_for_rc_channel(connection, center_x, center_y):
    while True:
        msg = connection.recv_match(type='RC_CHANNELS', blocking=True)
        channel_15_value = msg.chan15_raw  # Assuming raw value for channel 15 is what you're monitoring

        # Check if the value of channel 15 is above a certain threshold (like a button press or a threshold value)
        if channel_15_value > 1500:  # assuming 1500 is the threshold value when the button is pressed
            print(f"Channel 15 activated, executing script...")
            # Call the function to run the script with center_x and center_y
            run_program_after_delay(center_x, center_y)

        time.sleep(0.1)


    

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5n.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(320, 320),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    focal_length = 280,
    sensor_width = 3.2,
    sensor_height = 2.4,
    img_width = 640,
    img_height = 640,
    locked_target = None,
):
    
    rtsp_url = "rtsp://127.0.0.1:8554/live"
    fps = 30
    print("Frame rate of input video: ", fps)
    ffmpeg_command = [
        'ffmpeg',  # Read input at native frame rate
        '-y',  # Overwrite output files
        '-f', 'rawvideo',  # Input format
        '-vcodec', 'rawvideo',  # Codec
        '-pix_fmt', 'bgr24',  # Pixel format
        '-s', '640x480',  # Resolution
        '-r', str(fps),  # Frame rate from input video
        '-i', '-',  # Input from stdin
        '-an',  # No audio
        '-vcodec', 'libx264',  # Video codec
        '-preset', 'ultrafast',  # Adjust preset if needed
        '-tune', 'zerolatency',  # Minimize latency
        '-f', 'rtsp',  # RTSP format
        '-rtsp_transport', 'tcp',
        rtsp_url  # Output RTSP stream URL
    ]
    #ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt yuv420p -s 640x480 -r 30 -i - -an -vcodec libx264 -preset ultrafast -tune zerolatency -f rtsp -rtsp_transport tcp rtsp://127.0.2.1:8554/live

    min_distance = float('inf')  # Initialize to a very large value
    process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
    #prev_time = time.time()
    #elapsed_time = time.time() - prev_time


    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV fil

        #im0s = np.ascontiguousarray(im0s)
        #annotator = Annotator(im0s.copy(), line_width=line_thickness, example=str(names))
        #im0 = annotator.result()  # Assuming annotator.result() is the frame to be streamed
        #process.stdin.write(im0.tobytes())
        
        def calculate_camera_angles(center_x, center_y, img_width, img_height):
            dx = (center_x - img_width / 2) / (img_width / 2)  # Normalize to [-1, 1]
            dy = (center_y - img_height / 2) / (img_height / 2)  # Normalize to [-1, 1]
            yaw = math.degrees(math.atan2(dy, dx))
            pitch = math.degrees(math.atan2(dy, 1))
            roll = 0 
            return roll, pitch, yaw


        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)
                
            
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det = det[det[:, 5] == 0]  #PERSON DETECTION LINE ONLY!!!!!!!!!!!!
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()


                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string  

                for *xyxy, conf, cls in reversed(det):
                    if names[int(cls)] == 'person':
                        x1, y1, x2, y2 = map(int, xyxy)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        width = x2 - x1
                        height = y2 - y1
                        area  = width * height
                        distance = math.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
                        if distance < min_distance:
                            min_distance = distance
                           #closest_person = (*xyxy, conf, cls)
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f"Pixel: ({center_x}, {center_y})"
                    pixel_value = im0[center_y, center_x].tolist() if 0 <= center_x < im0.shape[1] and 0 <= center_y < im0.shape[0] else None
                    #print(("pixel: hhhhhhhhhhhhhhhh" , pixel_value ))
                    label = f"{model.names[int(cls)]} {conf:.2f} | Pixel: {pixel_value}"
                    cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    #print(f'Pixel Value at Center: {pixel_value}')
                    print(f"Width: {width}, Height: {height}, Area: {area}, Distance: {distance:.2f}")
                    print(f"BBox: ({x1}, {y1}), ({x2}, {y2}) | Center: ({center_x}, {center_y})")
                    rc_thread = threading.Thread(target=listen_for_rc_channel, args=(connection, center_x, center_y))
                    rc_thread.start()
                    
        #def execute_script():
            #script_path = "localization.py"  
            #os.system(f"python3 {script_path}")
            #print("Listening for RC channel 15 activation...")

         


                # Write results
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    print(f'BBox: ({x1}, {y1}), ({x2}, {y2}) | Center: ({center_x}, {center_y})')

                    roll, pitch, yaw = calculate_camera_angles(center_x, center_y, img_width, img_height)
                    print(f"Camera Angles: Roll = {roll:.2f}°, Pitch = {pitch:.2f}°, Yaw = {yaw:.2f}°")

                    fov_x = 2 * math.atan((sensor_width / 2) / focal_length) * (180 / math.pi)  # Horizontal FOV in degrees
                    fov_y = 2 * math.atan((sensor_height / 2) / focal_length) * (180 / math.pi) 
                    print(f"Camera FOV: Horizontal = {fov_x:.2f}°, Vertical = {fov_y:.2f}°")

                    cv2.circle(im0, (center_x, center_y), 5, (0, 255, 0), -1)
                    label = f'{names[int(cls)]} {conf:.2f}'
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        if save_format == 0:
                            coords = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            )  # normalized xywh
                        else:
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()  # xyxy
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)
                    #detect_pixel_value(im0, x1, y1, x2, y2, model)

        



            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
                im0s = np.ascontiguousarray(im0s)
                annotator = Annotator(im0s.copy(), line_width=line_thickness, example=str(names))
                im0 = annotator.result()  # Assuming annotator.result() is the frame to be streamed
                process.stdin.write(im0.tobytes())

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    #results = model(frame)
    #boxes = results.xyxy[0].cpu().numpy()
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    



    # Camera intrinsic parameters (replace with actual values)
focal_length = 800  # Example value (in pixels)
sensor_width = 6.3  # Example value (in mm)
sensor_height = 4.7  # Example value (in mm)

# Camera resolution (replace with actual resolution)
img_width = 640  # Image width (in pixels)
img_height = 640  # Image height (in pixels)

#for centre close person detection
frame_center_x = img_width // 2
frame_center_y = img_height // 2
min_distance = float('inf')
closest_person = None


# Calculate the Field of View (FOV)
fov_x = 2 * math.atan((sensor_width / 2) / focal_length) * (180 / math.pi)  # Horizontal FOV in degrees
fov_y = 2 * math.atan((sensor_height / 2) / focal_length) * (180 / math.pi)  # Vertical FOV in degrees

print(f"Camera FOV: Horizontal = {fov_x:.2f}°, Vertical = {fov_y:.2f}°")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5n.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default = "0", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):

    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
