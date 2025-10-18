from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

util_py = "/home/cxx/HWs/CS290U/project1/refs/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
weights = "/home/cxx/HWs/CS290U/project1/refs/GroundingDINO/models/groundingdino_swint_ogc.pth"


model = load_model(util_py, 
                   weights)
IMAGE_PATH = "/home/cxx/HWs/CS290U/project1/refs/GroundingDINO/assets/cat_dog.jpeg"
TEXT_PROMPT = "chair . person . dog ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("/home/cxx/HWs/CS290U/project1/refs/GroundingDINO/outputs/annotated_image.jpg", annotated_frame)
print("[OUT] annotated_image.jpg saved to outputs directory")