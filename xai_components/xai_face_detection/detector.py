from xai_components.base import InArg, OutArg, InCompArg, Component, xai_component
from deepface.detectors import FaceDetector as Detector
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import glob
import os
from pathlib import Path
import sys


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # current directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH


def detect_face(face_detector, detector_backend, img):
    """
    Detect face on input image with specified backend and return processed image and bounding box.

    Parameters:
        face_detector (model): Model built based on the backend type. Check out https://github.com/serengil/deepface/tree/master/deepface/detectors.
        detector_backend (str): Type of detector backend.
        img (array): Image loaded from file.

    Returns:
        img (array): Image with bounding box(es) drawn.
        bounding_box (list): List of bouunding box with format [x, y, w, h] which (x, y) is the top left corner, w represents width and h represents height.
    """
    # detect face
    try:
        result = Detector.detect_faces(
            face_detector, detector_backend, img, align=False
        )
    except:  # if detected face shape is (0, 0) and alignment cannot be performed, this block will be run
        result = []

    # draw bounding box
    bounding_box = []
    for faces in result:
        face, bb = faces
        bounding_box.append(bb)

        [x1, y1, w, h] = bb
        img = cv2.rectangle(
            img, (x1, y1), (x1 + w, y1 + h), color=(255, 0, 0), thickness=2
        )
    return img, bounding_box


@xai_component()
class FaceDetector(Component):
    """Face detection with popular face detectors. It takes folder or image as input and detects face with speficied backend (default: `mtcnn`). If folder, result will be saved in `output` folder besides the same directory provided, else, result will be saved at the root directory here (Check the message printed out at the terminal).

    ##### inPorts:
    - img_path: Folder or image file.
    - backend: Select face detector type. Available options are 0-5 which represent 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface' and 'mediapipe'. You may select the detector with its index or name.
        Default: 3
    - save_img: Save image(s) with bounding box drawn if True.
        Default: True. 
    - save_txt: Save bounding box found into text file if True.
        Default: True.

    ##### outPorts:
    - bb: If `img_path` is folder, dictionary with filename as key and bounding box as value is returned. Else, list of bounding box is returned. Noted that bounding box are in the format of [x, y, w, h] which (x, y) is the top left corner, w represents width and h represents height.
    """
    

    img_path: InCompArg[str]
    backend: InArg[any]
    save_img: InArg[bool]
    save_txt: InArg[bool]
    bb: OutArg[any]

    def __init__(self):

        self.done = False
        self.img_path = InCompArg(None)
        self.backend = InArg(3)
        self.save_img = InArg(True)
        self.save_txt = InArg(True)
        self.bb = OutArg(None)

    def execute(self, ctx) -> None:
        # backend
        backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"]
        try:
            detector_backend = backends[int(self.backend.value)]
        except:
            detector_backend = backends[self.backend.value]
        face_detector = Detector.build_model(detector_backend)

        # check input type and detect face
        print(f"Detection backend is '{detector_backend}'.")
        print("Input:", self.img_path.value)

        print("\nDetection start...")
        print(
            "Noted that bounding box are in the format of [x, y, w, h] which (x, y) is the top left corner, w represents width and h represents height."
        )

        # If input is image file
        if self.img_path.value.endswith((".jpg", ".png", ".jpeg")):

            # read input
            img = cv2.imread(self.img_path.value)

            # detection
            img, bounding_box = detect_face(face_detector, detector_backend, img)
            print("Bounding box found:", bounding_box)

            # show image
            img2 = cv2.cvtColor(
                img, cv2.COLOR_BGR2RGB
            )  # Converts from one colour space to the other
            plt.imshow(img2)
            plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
            plt.show()

            print("\nDone!")

            # save image
            if self.save_img.value:
                cv2.imwrite("output.jpg", img)
                print("Output image file saved to 'output.jpg'.")

            # save text
            if self.save_txt.value:
                with open("output.txt", "w") as f:
                    f.write(self.img_path.value + ": " + str(bounding_box))
                print("Bounding box saved to 'output.txt'.")
            self.bb.value = bounding_box

        # If input is folder
        else:
            bb = {}
            # configure filepath to save images
            path_list = os.path.normpath(self.img_path.value).split(os.sep)
            path_list.insert(-1, "output")
            output_folder = os.path.join(*path_list)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # open text file to write
            if self.save_txt.value:
                f = open(os.path.join(output_folder, "output.txt"), "w")

            # loop through all image files
            for file in tqdm(
                [
                    f
                    for f in glob.glob(self.img_path.value + "/**/*", recursive=True)
                    if f.endswith((".jpg", "png", "jpeg"))
                ]
            ):
                # read input
                img = cv2.imread(file)

                # detection
                print("\nProcessing", file, "...")
                img, bounding_box = detect_face(face_detector, detector_backend, img)
                print("Bounding box found:", bounding_box)
                bb[file] = bounding_box

                # show image
                img2 = cv2.cvtColor(
                    img, cv2.COLOR_BGR2RGB
                )  # Converts from one colour space to the other
                plt.imshow(img2)
                plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
                plt.show()

                # save result
                output_path = file.replace(self.img_path.value, output_folder)
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))
                if self.save_img.value:
                    cv2.imwrite(output_path, img)
                if self.save_txt.value:
                    f.write(file + ": " + str(bounding_box) + "\n")
            f.close()
            print("\nDone!")
            if self.save_img.value:
                print("Output image file saved to", str(output_folder) + ".")
            if self.save_txt.value:
                print(
                    "Bounding box saved to",
                    str(os.path.join(output_folder, "output.txt")) + ".",
                )
            self.bb.value = bb

        self.done = True


@xai_component(color="orange")
class CamInference(Component):
    
    """Camera inference with popular face detectors. 


    ##### inPorts:
    - backend: Select face detector type. Available options are 0-5 which represent 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface' and 'mediapipe'. You may select the detector with its index or name.
        Default: 3.
    - camera (int): If you have only one camera, default will mostly get the right camera. If you wish to use different camera, you can find your camera with different index such as 1, 2, 3, etc.
        Default: 0.
    - show_fps: Display FPS on the frame captured.
        Default: True.
    - show_count: Display face count on the frame captured.
        Default: True. 


    ##### outPorts:
    - bb: If `img_path` is folder, dictionary with filename as key and bounding box as value is returned. Else, list of bounding box is returned. Noted that bounding box are in the format of [x, y, w, h] which (x, y) is the top left corner, w represents width and h represents height.
    """

    backend: InArg[any]
    camera: InArg[int]
    show_fps: InArg[bool]
    show_count: InArg[bool]

    def __init__(self):
        self.done = False
        self.backend = InArg(3)
        self.camera = InArg(0)
        self.show_fps = InArg(True)
        self.show_count = InArg(True)

    def execute(self, ctx) -> None:
        # backend
        backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"]
        try:
            detector_backend = backends[int(self.backend.value)]
        except:
            detector_backend = backends[self.backend.value]
        face_detector = Detector.build_model(detector_backend)
        print(f"Detection backend is '{detector_backend}'.")

        cap = cv2.VideoCapture(self.camera.value)
        pTime = 0

        print("Press 'ESC' to exit.")
        print("Camera started!")
        while True:
            # read frame from camera
            success, img = cap.read()
            if not success:
                break

            # recolor to RGB for prediction
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # detection
            img, bounding_box = detect_face(face_detector, detector_backend, img)

            # display result
            if self.show_fps.value:
                pTime = display_framerate(pTime, img)
            if self.show_count.value:
                cv2.putText(
                    img,
                    "No. of face found: " + str(len(bounding_box)),
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # recolor to BGR for display
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Face Detection", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        self.done = True


def display_framerate(pTime, image):
    """
    Display framerate(FPS) on input image.

    Args:
        pTime (time.time): The time we start this iteration. FPS will be calculated based on this argument and the current time.
        image (array): Loaded image.

    Returns:
        pTime (time.time): Updated time to calculate FPS for next iteration.
    """
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        image,
        f"FPS: {int(fps)}",
        (image.shape[1] - 130, 40),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (255, 0, 0),
        2,
    )
    return pTime


@xai_component(type="utils")
class Print(Component):
    """
    Print the input.
    Parameters:
        item (any): Item to print out.
    """

    item: InCompArg[any]

    def __init__(self):
        self.done = False
        self.item = InCompArg(None)

    def execute(self, ctx) -> None:
        print(str(self.item.value))
        self.done = True
