# CNN Face Recognition / Face Detection

This repository contains a small, self-contained PyTorch **face detector**. It
does not identify who a person is. Instead, it predicts one or more bounding
boxes around faces and can draw those boxes on frames from the default webcam.

The detector is a grid-based convolutional neural network inspired by the
single-stage layout used by YOLO-style detectors:

1. An image is converted to RGB, normalized to `[0, 1]`, and letterboxed into
   a `128 x 128` square without stretching its aspect ratio.
2. A CNN extracts features and downsamples the image to an `8 x 8` grid.
3. Every grid cell predicts:
   - a face confidence logit;
   - the face-center offset inside that cell;
   - the face width and height.
4. During inference, predictions below a confidence threshold are discarded.
   Greedy non-maximum suppression (NMS) removes duplicate boxes.
5. The remaining boxes are mapped back from the letterboxed image to the
   original camera-frame coordinates and displayed with OpenCV.

The same letterbox and box-coordinate transformations are shared by training
and inference. This is important: using a different resize geometry at
inference time would make the model see a different coordinate system from the
one it learned during training.

## Repository layout

```text
CNN_model_face_recog/
|-- face_tracker.py
|-- requirements.txt
|-- README.md
|-- readme.txt                         # old placeholder file
`-- Model/
    |-- Model/
    |   |-- architecture.py
    |   |-- loss.py
    |   |-- preprocessing.py
    |   |-- model_training.py
    |   `-- face_detector.pt            # generated/bundled checkpoint
    `-- Images/
        |-- prepare_widerface_dataset.py
        |-- train/
        |   |-- images/
        |   `-- labels/
        |-- valid/
        |   |-- images/
        |   `-- labels/
        |-- wider_face_split/            # WIDER FACE annotation files
        |-- WIDER_train/                 # raw WIDER FACE training images
        |-- WIDER_val/                   # raw WIDER FACE validation images
        `-- WIDER_test/                  # raw WIDER FACE test images
```

The `Images` directory contains image and annotation data in addition to the
Python code. The raw WIDER FACE directories are inputs to the conversion
script. The `train` and `valid` directories are the format consumed by
`FaceDataset`. Python `__pycache__` directories and compiled `.pyc` files are
runtime artifacts and are not part of the application logic.

## Installation

Python 3.10 or newer is recommended because the code uses modern type
annotations such as `list[...]` and `Path | None`.

From PowerShell at the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependencies are:

| Package | Purpose |
| --- | --- |
| `torch` | CNN definition, tensors, automatic differentiation, optimizer, and checkpoint loading |
| `numpy` | Image arrays and preprocessing arrays |
| `Pillow` | Image loading, resizing, and WIDER FACE image inspection |
| `opencv-python` | Webcam capture, drawing, and display |
| `boto3` | Optional S3 dataset download support |

Install the PyTorch build appropriate for the machine if the default PyPI
build does not provide the desired CUDA support. The code automatically uses a
CUDA device when `torch.cuda.is_available()` is true; otherwise it uses the
CPU.

## Running the webcam detector

The bundled checkpoint is located at
`Model\Model\face_detector.pt`. Start the detector from the repository root:

```powershell
python .\face_tracker.py
```

The camera window shows a green box and a confidence score for every retained
face. Press `q` or `Esc` to stop.

### Inference options

```text
--camera CAMERA
    OpenCV camera index. Default: 0.

--model PATH
    Checkpoint to load. Default: Model\Model\face_detector.pt.

--threshold VALUE
    Minimum confidence in [0, 1]. Default: 0.5.
    Increase it to reduce false positives, at the cost of missed faces.

--nms-iou VALUE
    IoU threshold for merging overlapping predictions. Default: 0.4.

--nms-center-distance VALUE
    Relative center-distance threshold used to merge nearby duplicate
    predictions. Default: 0.5.
```

Examples:

```powershell
python .\face_tracker.py --camera 1
python .\face_tracker.py --model .\Model\Model\face_detector.pt --threshold 0.65
python .\face_tracker.py --nms-iou 0.35 --nms-center-distance 0.4
```

### Inference sequence

`face_tracker.py` performs the following operations for every camera frame:

1. `load_model` verifies that the checkpoint exists and contains a
   `model_state_dict`, constructs `CustomFaceDetector`, loads the weights, and
   switches the model to evaluation mode.
2. `predict_faces` computes letterbox parameters from the original frame
   dimensions.
3. OpenCV resizes the frame to the aspect-ratio-preserving dimensions and
   places it on a black `128 x 128` canvas.
4. BGR pixels are converted to RGB, divided by `255`, converted from
   `(height, width, channels)` to `(channels, height, width)`, and given a
   batch dimension.
5. The model returns a confidence-logit tensor with shape
   `(batch, 1, 8, 8)` and a box tensor with shape `(batch, 4, 8, 8)`.
6. Confidence logits are passed through `sigmoid`.
7. `decode_grid_predictions` converts each cell above the threshold into a
   normalized `[center_x, center_y, width, height]` box on the letterboxed
   canvas.
8. `non_max_suppression` removes duplicate detections.
9. `box_from_letterboxed` reverses the padding and scale transformation.
10. Coordinates are clipped to the camera frame and drawn with
    `cv2.rectangle` and `cv2.putText`.

The camera is always released and OpenCV windows are destroyed in the
`finally` block, including when the loop exits because of an exception.

## Dataset format

The training code expects this directory structure:

```text
dataset-root/
|-- train/
|   |-- images/
|   |   |-- image_a.jpg
|   |   `-- image_b.jpg
|   `-- labels/
|       |-- image_a.txt
|       `-- image_b.txt
`-- valid/
    |-- images/
    `-- labels/
```

Every image used by `FaceDataset` must have a label file with the same stem.
Each non-empty label line uses normalized YOLO-style coordinates:

```text
class_id center_x center_y width height
```

For this project `class_id` is always `0`, because the only class is “face”.
All coordinates are normalized to the **original image**, not to pixels:

```text
center_x = (left + box_width / 2) / image_width
center_y = (top  + box_height / 2) / image_height
width    = box_width  / image_width
height   = box_height / image_height
```

An empty label file represents an image with no usable face. Such negative
images are retained by the dataset and provide negative confidence examples.
The loader also accepts a line containing more than five values as a normalized
polygon: the points after the class ID are converted to the enclosing
`[center_x, center_y, width, height]` box.

### Preparing WIDER FACE data

`Model\Images\prepare_widerface_dataset.py` converts WIDER FACE annotation
text files into the structure above. It copies images without cropping them and
writes one normalized label line per valid face.

Example commands from the repository root:

```powershell
python .\Model\Images\prepare_widerface_dataset.py `
  --wider-root .\Model\Images\WIDER_train\images `
  --annotations .\Model\Images\wider_face_split\wider_face_train_bbx_gt.txt `
  --output .\Model\Images\train

python .\Model\Images\prepare_widerface_dataset.py `
  --wider-root .\Model\Images\WIDER_val\images `
  --annotations .\Model\Images\wider_face_split\wider_face_val_bbx_gt.txt `
  --output .\Model\Images\valid
```

The converter:

1. Parses each WIDER FACE image record and its face count.
2. Reads `(x, y, width, height)` pixel boxes.
3. Drops boxes marked invalid by the WIDER annotation and boxes with
   non-positive dimensions.
4. Drops faces smaller than `--min-face-px` (default `16`), because an `8 x 8`
   grid has limited localization resolution for extremely small faces.
5. Skips an image if it originally had faces but all of them were removed by
   the size filter. Keeping that image as a negative would incorrectly teach
   the detector that visible, merely tiny faces are background.
6. Copies the source image and writes its label file.

The script's `parse_annotations` function also handles WIDER's zero-face
records, which contain a dummy annotation line.

## How training works

Training is implemented in
`Model\Model\model_training.py` and is started with:

```powershell
python .\Model\Model\model_training.py
```

By default it reads `Model\Images\train` and `Model\Images\valid`, trains for
30 epochs with batch size 32 and learning rate `0.001`, and writes the best
checkpoint to `Model\Model\face_detector.pt`.

### Training options

```text
--data PATH_OR_S3_URI
    Local dataset root or an S3 URI. Default: Model\Images.

--s3-cache PATH
    Local cache used when --data is an s3:// URI.
    Default: ~/.cache/cnn_model_face_recog.

--epochs INTEGER
    Number of complete passes over the training set. Default: 30.

--batch-size INTEGER
    Number of images per optimizer update. Default: 32.

--learning-rate FLOAT
    Adam learning rate. Default: 0.001.

--output PATH
    Output checkpoint path. Default: Model\Model\face_detector.pt.
```

Example:

```powershell
python .\Model\Model\model_training.py `
  --data .\Model\Images `
  --epochs 50 `
  --batch-size 16 `
  --learning-rate 0.001 `
  --output .\Model\Model\face_detector.pt
```

S3 training is supported with a URI such as
`s3://bucket/prefix`. `download_s3_dataset` lists the prefix with a paginated
S3 client and downloads only files below `train/images`, `train/labels`,
`valid/images`, or `valid/labels`. Files already present with the same size are
treated as cached. AWS credentials must already be available to `boto3`; the
repository does not contain credentials.

### 1. Loading and transforming one sample

`FaceDataset.__getitem__` loads an image and:

1. Converts it to RGB.
2. Computes a letterbox transform with `compute_letterbox_params`.
3. Resizes it while preserving aspect ratio.
4. Pastes it into a black square using `paste_into_canvas`.
5. Divides pixel values by `255`, producing floats in `[0, 1]`.
6. Reads every label associated with the image.
7. Converts each original-image box to letterboxed coordinates with
   `box_to_letterboxed`.
8. When `training=True`, horizontally flips the image and updates every box's
   center with `center_x = 1 - center_x` with probability `0.5`.
9. Encodes all boxes into an `8 x 8` target grid with
   `encode_grid_targets`.
10. Returns:
    - image: `(3, 128, 128)` `torch.Tensor`;
    - confidence target: `(1, 8, 8)` float tensor;
    - box target: `(4, 8, 8)` float tensor.

The `DataLoader` adds the batch dimension, so the model receives
`(batch, 3, 128, 128)`.

### 2. Grid target encoding

The `128 x 128` input is divided into `8 x 8` cells. Each cell therefore
covers `16 x 16` pixels. For a normalized face center `(cx, cy)`:

```text
grid_x = int(cx * 8)
grid_y = int(cy * 8)
offset_x = cx * 8 - grid_x
offset_y = cy * 8 - grid_y
```

The confidence target for that cell becomes `1`, and the box target stores:

```text
[offset_x, offset_y, normalized_width, normalized_height]
```

All other confidence targets are `0`. Only one face can occupy one cell. If
two face centers map to the same cell, the first box is kept and the later box
is dropped. This is a deliberate YOLO-style simplification and is the main
resolution limitation of the current detector.

### 3. CNN forward pass

`CustomFaceDetector` contains four convolution/max-pooling stages:

| Stage | Operation | Output shape for one image |
| --- | --- | --- |
| Input | RGB image | `3 x 128 x 128` |
| 1 | `3 -> 16` convolution, ReLU, `2 x 2` max pool | `16 x 64 x 64` |
| 2 | `16 -> 32` convolution, ReLU, max pool | `32 x 32 x 32` |
| 3 | `32 -> 64` convolution, ReLU, max pool | `64 x 16 x 16` |
| 4 | `64 -> 128` convolution, ReLU, max pool | `128 x 8 x 8` |
| Head | `1 x 1` convolution, `128 -> 5` channels | `5 x 8 x 8` |

The five head channels are split into:

```text
channel 0: confidence logits
channels 1-4: offset_x, offset_y, width, height
```

The confidence output intentionally remains a logit because
`binary_cross_entropy_with_logits` is numerically stable. The four box
channels receive `sigmoid` in `forward`, constraining them to `[0, 1]`.

`GRID_SIZE` is defined once in `preprocessing.py` as `IMAGE_SIZE // 16` and
imported by the architecture. The assertion in `forward` catches a mismatch
between the convolutional downsampling and the target encoding.

### 4. Loss function

`FaceDetectionLoss` combines confidence loss and box regression loss:

```text
total_loss = confidence_loss + 5.0 * box_loss
```

The confidence term is binary cross entropy with logits. Empty cells use a
weight of `0.5`, while cells containing faces use a weight of `1.0`. This
reduces the effect of the many easy negative cells without ignoring them.

The box term is mean squared error, but is computed only at positive cells.
It is summed over the four box channels and divided by the number of faces in
the batch (with a minimum denominator of `1` to avoid division by zero).
The default weights are:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `noobj_weight` | `0.5` | Weight of an empty-cell confidence target |
| `obj_weight` | `1.0` | Weight of a face-cell confidence target |
| `box_weight` | `5.0` | Relative importance of box regression |

### 5. Epoch and checkpoint behavior

`run_epoch` handles both modes:

- If an optimizer is supplied, it enables training mode, clears gradients,
  performs the forward pass, backpropagates, and calls `optimizer.step()`.
- If no optimizer is supplied, it enables evaluation mode and only computes
  loss.

`main` creates an Adam optimizer, runs the training epoch, then runs a
no-gradient validation epoch. It prints the mean training and validation loss.
Only the checkpoint with the lowest validation loss is saved:

```python
{
    "model_state_dict": model.state_dict(),
    "valid_loss": valid_loss,
}
```

This is why inference expects the `model_state_dict` key rather than loading a
serialized model object.

## Source-file reference

### [`face_tracker.py`](./face_tracker.py)

Webcam inference entry point.

- `load_model(model_path, device)` validates and loads a checkpoint into
  `CustomFaceDetector`.
- `predict_faces(model, frame, device, ...)` applies the inference transform,
  runs the network, decodes grid predictions, performs NMS, converts boxes back
  to frame pixels, and returns `(confidence, coordinates)` tuples.
- `main()` parses command-line options, selects CPU or CUDA, opens the camera,
  runs the display loop, draws boxes, and cleans up OpenCV resources.

### [`Model/Model/architecture.py`](./Model/Model/architecture.py)

Defines the neural network.

- `CustomFaceDetector.__init__()` creates the four convolution layers, shared
  max-pooling layer, and per-grid-cell `1 x 1` detection head.
- `CustomFaceDetector.forward(x)` produces confidence logits and sigmoid-box
  predictions in grid layout.

### [`Model/Model/preprocessing.py`](./Model/Model/preprocessing.py)

The single source of truth for image geometry and grid conversion.

- `LetterboxParams` stores scale, resized dimensions, and horizontal/vertical
  padding.
- `compute_letterbox_params(width, height, target_size)` computes an
  aspect-ratio-preserving resize.
- `paste_into_canvas(resized, params, target_size)` puts a resized image on a
  black square canvas.
- `box_to_letterboxed(box, width, height, params, target_size)` maps normalized
  original-image coordinates into the letterboxed canvas.
- `box_from_letterboxed(box, width, height, params, target_size)` reverses that
  mapping into original-frame pixel coordinates.
- `encode_grid_targets(boxes, grid_size)` assigns face boxes to cells and
  creates confidence and regression targets.
- `decode_grid_predictions(confidence, box, grid_size, confidence_threshold)`
  turns above-threshold cell outputs back into normalized boxes.
- `_iou(box_a, box_b)` calculates intersection over union for two center-size
  boxes.
- `_center_distance_ratio(box_a, box_b)` measures center separation relative
  to the boxes' average size.
- `non_max_suppression(detections, iou_threshold,
  center_distance_ratio_threshold)` greedily retains the highest-confidence
  boxes and removes boxes that overlap or have very close centers.

### [`Model/Model/loss.py`](./Model/Model/loss.py)

- `FaceDetectionLoss.__init__(noobj_weight, obj_weight, box_weight)` stores
  confidence and box-loss weights.
- `FaceDetectionLoss.forward(pred_conf, pred_box, target_conf, target_box)`
  computes weighted confidence BCE plus positive-cell box MSE.

### [`Model/Model/model_training.py`](./Model/Model/model_training.py)

Training data and the command-line training program.

- `download_s3_dataset(s3_uri, cache_root)` validates an S3 URI and downloads
  the expected train/validation tree into a local cache.
- `FaceDataset.__init__(split_dir, image_size, training)` indexes images that
  have matching label files.
- `FaceDataset.__len__()` returns the number of indexed samples.
- `FaceDataset.__getitem__(index)` loads, letterboxes, optionally flips, and
  encodes one sample.
- `run_epoch(model, dataloader, criterion, device, optimizer)` executes one
  training or validation pass and returns its mean loss.
- `main()` parses options, optionally downloads data, creates data loaders,
  initializes the model/loss/optimizer, runs epochs, validates, and saves the
  best checkpoint.

### [`Model/Images/prepare_widerface_dataset.py`](./Model/Images/prepare_widerface_dataset.py)

WIDER FACE conversion utility.

- `parse_annotations(annotation_path)` parses WIDER FACE records and returns
  valid pixel-coordinate boxes keyed by image path.
- `main()` parses conversion options, filters invalid/small faces, copies
  images, writes normalized labels, and reports saved/skipped counts.

### [`requirements.txt`](./requirements.txt)

Lists the runtime packages required for training, preprocessing, webcam
inference, and optional S3 access.

### [`.gitignore`](./.gitignore)

Contains repository ignore rules. The dataset directory is ignored so large
image collections do not need to be added as ordinary source files.

### [`readme.txt`](./readme.txt)

An old placeholder file. This `README.md` is the canonical project
documentation.

## Important limitations and practical considerations

- This is face **detection**, not face recognition or identity classification.
- The model predicts at most one face per grid cell. Very close or very small
  faces can collide in the same `16 x 16` cell.
- The default input is only `128 x 128`, so tiny faces and fine localization
  are inherently difficult.
- Training reports loss only. There is no built-in precision, recall, IoU,
  mAP, or per-face validation metric.
- The checkpoint is saved only when validation loss improves; interrupted
  training does not resume optimizer state or epoch state.
- Camera capture and display require a graphical desktop and a working OpenCV
  camera backend.
- S3 access requires AWS credentials configured outside this repository.
- Changing `IMAGE_SIZE`, the number of pooling stages, or `GRID_SIZE` requires
  keeping the architecture, target encoder, and inference code consistent.

## End-to-end summary

```text
WIDER FACE annotations
        |
        v
prepare_widerface_dataset.py
        |
        v
train/images + train/labels
valid/images + valid/labels
        |
        v
FaceDataset
  letterbox + normalize + optional flip
  box_to_letterboxed + encode_grid_targets
        |
        v
CustomFaceDetector
  128x128 -> CNN -> 8x8 confidence/box grid
        |
        v
FaceDetectionLoss
  weighted confidence BCE + positive-cell box MSE
        |
        v
Adam updates, validation, best checkpoint
        |
        v
face_tracker.py
  same letterbox -> prediction -> sigmoid -> decode -> NMS
  -> inverse letterbox -> boxes drawn on webcam frames
```
