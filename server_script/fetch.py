import pygame
import io
from PIL import Image
from datetime import datetime, timedelta
from minio import Minio

# Initialize MinIO client
client = Minio(
    "10.10.1.110:9000",
    access_key="xtcZX7OoGoqxRfsc4eE5",
    secret_key="ZSHixcePInx7NQEY74Bwpkwf4Y5DqD23WWzTOT20",
    secure=False
)

# Define bucket and date/time to start fetching images
BUCKET_NAME = "security-camera"
START_TIME = datetime.now() - timedelta(seconds=60)

# Initialize Pygame
pygame.init()
WINDOW_SIZE = (324, 324)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption('MinIO Image Playback')
time_font = pygame.font.Font(None, 22)


def extract_timestamp_from_filename(filename):
    """Extract timestamp from filename."""
    try:
        timestamp_str = filename[len("img_stream_"):].split('.')[0]

        print(timestamp_str)
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")
    except (IndexError, ValueError):
        return None


def fetch_images_from_minio():
    """Fetch and filter image filenames from MinIO bucket based on the timestamp."""
    images = []
    filenames = []
    print("Fetching images starting with: " + "img_stream_" + START_TIME.strftime("%Y%m%d_%H%M%S_"))
    for obj in client.list_objects(BUCKET_NAME, recursive=True,
                                   start_after="img_stream_" + START_TIME.strftime("%Y%m%d_%H%M%S_")):
        try:
            obj_data = client.get_object(BUCKET_NAME, obj.object_name)
            img_data = io.BytesIO(obj_data.read())
            images.append(img_data)
            filenames.append(obj.object_name)
        except Exception as e:
            print(f"Error fetching object {obj.object_name}: {e}")

    # Sort images and filenames based on timestamps
    sorted_indices = sorted(range(len(filenames)), key=lambda i: extract_timestamp_from_filename(filenames[i]))
    sorted_images = [images[i] for i in sorted_indices]
    sorted_filenames = [filenames[i] for i in sorted_indices]

    return sorted_images, sorted_filenames


def draw_system_time(surface):
    """Draw the current system time on the surface."""
    current_time = datetime.now().strftime("%Y-%m-%d %A %H:%M:%S")
    time_surface = time_font.render(current_time, True, (255, 255, 255))
    surface.blit(time_surface,
                 (WINDOW_SIZE[0] - time_surface.get_width() - 10, WINDOW_SIZE[1] - time_surface.get_height() - 10))


def play_images_as_video(images, filenames):
    """Play images as video with respect to the relative time difference between frames."""
    if not images or not filenames:
        return

    # Calculate frame delays based on timestamps
    frame_delays = []
    timestamps = [extract_timestamp_from_filename(f) for f in filenames]

    for i in range(1, len(timestamps)):
        time_diff = (timestamps[i] - timestamps[i - 1]).total_seconds() * 1000  # Convert to milliseconds
        frame_delays.append(max(int(time_diff), 0))  # Ensure non-negative delay

    # Play images
    for i, img_data in enumerate(images):
        try:
            img = Image.open(img_data)
            img = img.convert('RGB')
            img = img.resize(WINDOW_SIZE)
            img_surface = pygame.image.fromstring(img.tobytes(), img.size, 'RGB')
            screen.blit(img_surface, (0, 0))
            draw_system_time(screen)
            pygame.display.update()

            # Wait for the appropriate amount of time for the current frame
            if i < len(frame_delays):
                pygame.time.wait(frame_delays[i])
        except Exception as e:
            print(f"Error loading image: {e}")


def main():
    images, filenames = fetch_images_from_minio()
    if images:
        play_images_as_video(images, filenames)
    else:
        print("No images found from the specified time.")

    # Main event loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()


if __name__ == "__main__":
    main()
