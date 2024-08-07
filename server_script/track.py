"""
This script receives TCP image stream, multi-object tracking and uploads the results to minio DB.
"""
import numpy as np
import pygame
import socket
from pygame.locals import QUIT
import io
from PIL import Image
import json
from datetime import datetime
from minio import Minio

from bytetrack.byte_tracker import BYTETracker


np.float = float  # workaround for tracker

motion_buffer = []
MOTION_BUFFER_SIZE = 5


# Initialize MinIO client
client = Minio(
    "<IP>",
    access_key="",
    secret_key="",
    secure=False
)

# Define coral board IP and port
TCP_IP = ""
TCP_PORT = 31337

# Tracker settigns
frame_id = 0
results = []

aspect_ratio_thresh = 0.9
min_box_area = 0.0
track_thresh = 0.5
track_buffer = 64
match_thresh = 1.0
fuse_score = False
frame_rate = 2.0
tracker = BYTETracker(
    track_thresh=track_thresh,
    track_buffer=track_buffer,
    match_thresh=match_thresh,
    fuse_score=fuse_score,
    frame_rate=frame_rate)

# Pygame settings
IMG_SIZE = [324, 324]
WINDOW_SIZE = (324, 324)
screen = pygame.display.set_mode(WINDOW_SIZE)



def filter_motion_detection(motion_detected):
    """Filter motion detection results using a history of states."""
    global motion_buffer

    # Add the current motion state to the buffer
    if len(motion_buffer) >= MOTION_BUFFER_SIZE:
        motion_buffer.pop(0)
    motion_buffer.append(motion_detected)

    # Determine if motion is detected based on the majority vote
    if len(motion_buffer) > 0:
        motion_vote = np.mean(motion_buffer)
        return motion_vote > 0.5
    return False


def connect_to_server():
    """Create a new socket connection to the server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((TCP_IP, TCP_PORT))
    return sock


def draw_bounding_boxes(surface, bboxes, img_width, img_height):
    """Draw bounding boxes on the image"""
    for bbox in bboxes:
        xmin = int(bbox['xmin'] * img_width)
        ymin = int(bbox['ymin'] * img_height)
        xmax = int(bbox['xmax'] * img_width)
        ymax = int(bbox['ymax'] * img_height)

        pygame.draw.rect(surface, (255, 0, 0), pygame.Rect(xmin, ymin, xmax - xmin, ymax - ymin), 2)
        font = pygame.font.Font(None, 30)
        text = font.render(bbox['id'], True, (255, 0, 0))
        surface.blit(text, (xmin, ymin - 20))


def parse_bbox_data(bbox_data):
    """Parse the bounding box data from a list of strings."""
    bboxes = []
    for line in bbox_data:
        parts = line.strip().split(',')
        if len(parts) < 7:
            continue  # Skip any invalid data
        bbox = {
            'id': parts[1],
            'xmin': float(parts[2]) / 324,
            'ymin': float(parts[3]) / 324,
            'xmax': float(parts[4]) / 324 + float(parts[2]) / 324,
            'ymax': float(parts[5]) / 324 + float(parts[3]) / 324,
            'score': float(parts[6])
        }
        bboxes.append(bbox)
    return bboxes


def draw_system_time(surface):
    """Draw the current system time on the surface."""
    current_time = datetime.now().strftime("%Y-%m-%d %A %H:%M:%S")
    time_font = pygame.font.Font(None, 12)
    time_surface = time_font.render(current_time, True, (255, 255, 255))
    surface.blit(time_surface,
                 (WINDOW_SIZE[0] - time_surface.get_width() - 10, WINDOW_SIZE[1] - time_surface.get_height() - 10))


def save_frame_to_jpeg(surface):
    """Save the current Pygame surface to a JPEG file and upload to MinIO."""
    image_data = pygame.surfarray.array3d(surface)
    image = Image.fromarray(image_data.transpose(1, 0, 2))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=95)
    img_byte_arr.seek(0)
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"img_stream_{now}.jpg"
    result = client.put_object(
        "security-camera",
        filename,
        img_byte_arr,
        img_byte_arr.getbuffer().nbytes
    )
    print(
        "created {0} object; etag: {1}, version-id: {2}".format(
            result.object_name, result.etag, result.version_id,
        ))
    print(f"File uploaded successfully: {result}")


def draw_green_border(surface):
    """Draw a green border around the image if motion is detected."""
    pygame.draw.rect(surface, (0, 255, 0), pygame.Rect(0, 0, WINDOW_SIZE[0], WINDOW_SIZE[1]), 5)


def convert_json_to_array(json_data):
    # Parse JSON data
    data = json.loads(json_data)

    # Extract bounding boxes and scores
    bboxes = data.get('bboxes', [])
    num_boxes = len(bboxes)

    # Initialize the NumPy array with shape (num_boxes, 5)
    output_results = np.zeros((num_boxes, 5))

    # Fill the array with bounding boxes and scores
    for i, bbox in enumerate(bboxes):
        output_results[i, 0] = bbox['xmin'] * 324
        output_results[i, 1] = bbox['ymin'] * 324
        output_results[i, 2] = bbox['xmax'] * 324
        output_results[i, 3] = bbox['ymax'] * 324
        output_results[i, 4] = bbox['score']

    return output_results


def receive_data(sock):
    """Receive and process data from the socket."""
    buffer = b''
    sop = b"<START_OF_PAYLOAD>"
    eop = b"<END_OF_PAYLOAD>"
    delimiter = b"<END_OF_IMAGE>"

    while True:
        try:
            chunk = sock.recv(8192)
            if not chunk:
                break
            buffer += chunk

            # Look for start marker
            if sop in buffer:
                start_index = buffer.find(sop) + len(sop)
                buffer = buffer[start_index:]

            # Look for end marker
            if eop in buffer:
                end_index = buffer.find(eop)
                payload = buffer[:end_index]
                buffer = buffer[end_index + len(eop):]

                # Process the JPEG and bbox data
                if delimiter in payload:
                    jpeg_data, bbox_data = payload.split(delimiter, 1)

                    # Process JPEG data
                    try:
                        img = Image.open(io.BytesIO(jpeg_data))
                        img = img.convert('RGB')
                        img = img.resize(WINDOW_SIZE)
                        img_surface = pygame.image.fromstring(img.tobytes(), img.size, 'RGB')
                        screen.blit(img_surface, (0, 0))
                        pygame.display.update()
                    except Exception as e:
                        print(f'Error processing image: {e}')

                    try:
                        bbox_json = json.loads(bbox_data.decode('utf-8'))
                        results.clear()
                        global frame_id
                        output_results = convert_json_to_array(bbox_data)
                        if output_results[0] is not None:
                            print(output_results)

                            online_targets = tracker.update(output_results, IMG_SIZE,
                                                            IMG_SIZE)

                            online_tlwhs = []
                            online_ids = []
                            online_scores = []
                            for t in online_targets:
                                tlwh = t.tlwh
                                tid = t.track_id
                                online_tlwhs.append(tlwh)
                                online_ids.append(tid)
                                online_scores.append(t.score)

                                results.append(
                                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                                )

                        frame_id += 1

                        draw_bounding_boxes(screen, parse_bbox_data(results), WINDOW_SIZE[0], WINDOW_SIZE[1])
                        #  draw_bounding_boxes(screen, bbox_json.get('bboxes', []), WINDOW_SIZE[0], WINDOW_SIZE[1])
                        is_motion_detected = bbox_json.get('isMotionDetected', 0)
                        if filter_motion_detection(is_motion_detected):
                            draw_green_border(screen)
                        pygame.display.update()
                    except json.JSONDecodeError:
                        print("Error decoding JSON.")

        except Exception as e:
            print(f'Error receiving data: {e}')
            break

    sock.close()


def main():
    # Initialize Pygame
    pygame.init()

    pygame.display.set_caption('TCP JPEG Image Stream')

    while True:
        sock = connect_to_server()
        sock.settimeout(1)
        receive_data(sock)
        draw_system_time(screen)
        pygame.display.update()
        #save_frame_to_jpeg(screen)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()


if __name__ == "__main__":
    main()
