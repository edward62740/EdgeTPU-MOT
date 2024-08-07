# Edge TPU Multi Object Tracking

This project uses a Coral Dev Micro board to perform on-device inference, and streams the inference results and compressed JPEGs to a server running track.py, which performs MOT and uploads the results to MinIO DB.

For the bounding box detector, either Yolov8n or MobileDet can be configured. However, Yolov8 runs about 50% slower due to some limitation of the TPU.

## Performance
The system runs inference and the TCP transmission tasks at around 4-5 FPS at medium TPU freq. In theory, it could run the model much faster without the additional processing steps. The system consumes about 1W.


## Future improvements
To speed up inference, it's possible to delegate the frame buffer and NMS tasks to the M4 processor, which is currently idle.
Also, the TPU appears to have a problem with the specific tensor shown below (for Yolov8, before the TRANSPOSE), but not for smaller sizes of similar operations of transpose followed by a reshape. It may be possible to modify the
model to slice the tensor before the transpose and perform N such operations before concatenating the outputs.
![Screenshot from 2024-08-07 09-30-33](https://github.com/user-attachments/assets/4b4e87ad-561d-45cc-bee1-31d417364ba5)
