# Xircuits Template - Face Detection 🚀
"""
Face detection with popular face detectors.
Usage:
    $ python main.py -i "image/img1.jpg" -b 4
    $ python main.py -i "image" -b 0

RetinaFace and MTCNN seem to overperform in detection and alignment stages but they are much slower. If the speed of your pipeline is more important, then you should use opencv or ssd. On the other hand, if you consider the accuracy, then you should use retinaface or mtcnn.
Code base: https://github.com/serengil/deepface
"""

from deepface.detectors import FaceDetector
import cv2
from tqdm import tqdm
import glob
import os
import argparse

# if you need dlib and can't download it with pip install,
# try https://github.com/datamagic2020/Install-dlib


def detect_face(face_detector, detector_backend, img):
    # detect face
    try:
        result = FaceDetector.detect_faces(
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--img", help="Your image path to detect face(s).", required=True
    )
    ap.add_argument(
        "-b",
        "--backend",
        help="Select face detector type. Available options are 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface' and 'mediapipe'(You can select the detector with its index or name).",
        default=4,
    )
    ap.add_argument(
        "--save_img",
        help="Save image(s) with bounding box drawn if True.",
        default=True,
    )
    ap.add_argument(
        "--save_text",
        help="Save filename and bounding box found into text file if True.",
        default=True,
    )
    args = vars(ap.parse_args())

    # backend
    backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"]
    try:
        detector_backend = backends[int(args["backend"])]
    except:
        detector_backend = backends[args["backend"]]
    face_detector = FaceDetector.build_model(detector_backend)

    # check input type and detect face
    print("Detection start...")
    print("Input: ", args["img"])
    if args["img"].endswith((".jpg", ".png", ".jpeg")):
        # read input
        img = cv2.imread(args["img"])
        # detection
        img, bounding_box = detect_face(face_detector, detector_backend, img)
        print("Bounding box found:", bounding_box)
        # save image
        if args["save_img"]:
            cv2.imwrite("output.jpg", img)
            print("Output image file saved to 'output.jpg'.")
        # save text
        if args["save_text"]:
            with open(args["img"].split(".")[0] + ".txt", "w") as f:
                f.write(args["img"] + ": " + str(bounding_box))
            print("Bounding box saved to", args["img"].split(".")[0] + ".txt")

    else:
        # configure filepath to save images
        path_list = os.path.normpath(args["img"]).split(os.sep)
        path_list.insert(-1, "output")
        output_folder = os.path.join(*path_list)
        print("Output images will be saved to", output_folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # open text file to write
        if args["save_text"]:
            f = open(os.path.join(output_folder, "output.txt"), "w")

        # loop through all image files
        for file in tqdm(
            [
                f
                for f in glob.glob(args["img"] + "/**/*", recursive=True)
                if f.endswith((".jpg", "png", "jpeg"))
            ]
        ):
            img = cv2.imread(file)
            print("\nProcessing", file, "...")
            img, bounding_box = detect_face(face_detector, detector_backend, img)
            print("Bounding box found:", bounding_box)

            # save result
            output_path = file.replace(args["img"], output_folder)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            if args["save_img"]:
                cv2.imwrite(output_path, img)
            if args["save_text"]:
                f.write(file + ": " + str(bounding_box) + "\n")
        f.close()
        print("Output image file saved to", str(output_folder) + ".")
        print(
            "Bounding box saved to",
            str(os.path.join(output_folder, "output.txt")) + ".",
        )
