# Xircuits Project Template

This template allows you to apply face detection on images.

It consists of 1 component:

- [FaceDetector](/xai_components/xai_face_detection/detector.py#L41): Face detection with popular face detectors. It takes folder or image as input and detects face with speficied backend (default: `mtcnn`). If folder, result will be saved in `output` folder besides the same directory provided, else, result will be saved at the root directory here (Check the message printed out at the terminal).

## Prerequisites

Python 3.9

## Installation

1. Clone this repository

```
git clone https://github.com/XpressAI/template-face_detection.git
```

2. Create a virtual environment and install the required python packages

```
pip install -r requirements.txt
```

3. Run xircuits from the root directory

```
xircuits
```

## Workflow in this Template

### Inference

#### [FaceDetector.xircuits](/xircuits-workflows/FaceDetector.xircuits)

<details>
<summary>Where to provide input?</summary>

Input is taken by `img_path` of Component `FaceDetector`. Check out the [example](#example) below.

You may provide folder or image file as input. Relative path or absolute path both are accepted (Noted that your folder in path should be separated by `/`).

</details>

<details>
<summary>Where to find output?</summary>

If your input is folder, the output folder will be located besides your input folder.
Example:
Input: `resource/sample`
Output: `resource/output/sample`

If your input is image file, the output will be located in this github root directory (the same directory as this README) named `output.jpg` and `output.txt`.

</details>

##### Example

![inference](/resource/image/inference.gif)

### Real-time Inference

#### [CamInference.xircuits](/xircuits-workflows/CamInference.xircuits)

##### Example

![cam_inference]()
