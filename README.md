# Edge TPU Multi Object Tracking

This project uses a Coral Dev Micro board to perform on-device inference, and streams the inference results and compressed JPEGs to a server running track.py, which performs MOT using ByteTrack and uploads the results to MinIO DB.<br>

For the bounding box detector, either Yolov8n or MobileDet can be configured. However, Yolov8 runs about 50% slower due to some limitation of the TPU.<br>

The device also performs motion detection, which can be used to trigger recording or some other actions.

## Performance
The system runs inference and the TCP transmission tasks at around 4-5 FPS at medium TPU freq. In theory, it could run the model much faster without the additional processing steps. The system consumes about 1W.<br>

## Future improvements
To speed up inference, it's possible to delegate the frame buffer and NMS tasks to the M4 processor, which is currently idle.
Also, the TPU appears to have a problem with the specific tensor shown below (for Yolov8, before the TRANSPOSE), but not for smaller sizes of similar operations of transpose followed by a reshape. It may be possible to modify the
model to slice the tensor before the transpose and perform N such operations before concatenating the outputs.<br>

![Screenshot from 2024-08-07 09-30-33](https://github.com/user-attachments/assets/4b4e87ad-561d-45cc-bee1-31d417364ba5)


## Citation

```
@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

