import cv2
import xml.etree.ElementTree as ET
import random
import os
from collections import defaultdict
from dataclasses import dataclass

#Estructura básica para el peatón
@dataclass
class Pedestrian:
    ped_id: str
    bounding_box: list  #[xtl, ytl, xbr, ybr]
    cross: str
    action: str


#Función para dibujar bounding boxes
def plot_one_box(pedestrian, img, color=None, line_thickness=2):
    x = pedestrian.bounding_box
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]

    c1 = (int(x[0]), int(x[1]))
    c2 = (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)


#Parseo de las anotaciones de JAAD
def load_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frame_dict = defaultdict(list)

    for track in root.findall(".//track[@label='pedestrian']"):
        for box in track.findall("box"):
            frame_id = int(box.attrib["frame"])

            attributes = {
                attr.attrib["name"]: attr.text
                for attr in box.findall("attribute")
            }

            ped = Pedestrian(
                ped_id=attributes.get("id", "unknown"),
                bounding_box=[
                    float(box.attrib["xtl"]),
                    float(box.attrib["ytl"]),
                    float(box.attrib["xbr"]),
                    float(box.attrib["ybr"]),
                ],
                cross=attributes.get("cross", "not-crossing"),
                action=attributes.get("action", "unknown"),
            )

            frame_dict[frame_id].append(ped)

    return frame_dict


#Dibujado de las bounding boxes
def draw_bboxes_on_video(
    video_path,
    annotations,
    output_dir="output_frames",
    max_frames=50
):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in annotations:
            for ped in annotations[frame_idx]:
                color = (0, 255, 0) if ped.cross == "crossing" else (0, 0, 255)
                plot_one_box(pedestrian=ped, img=frame, color=color)

                label = f"{ped.ped_id} | {ped.cross}"
                cv2.putText(
                    frame,
                    label,
                    (int(ped.bounding_box[0]), int(ped.bounding_box[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA
                )

            out_path = f"{output_dir}/frame_{frame_idx}.jpg"
            cv2.imwrite(out_path, frame)
            saved += 1

            if saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"Frames guardados: {saved}")


#Main
if __name__ == "__main__":
    XML_PATH = "video_0215.xml"
    VIDEO_PATH = "video_0215.mp4"

    annotations = load_annotations(XML_PATH)

    draw_bboxes_on_video(
        video_path=VIDEO_PATH,
        annotations=annotations,
        output_dir="output_frames",
        max_frames=500
    )
