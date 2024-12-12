import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load YOLO model
model = load_model("model_final.h5")

# Class dictionary
class_dictionary = {0: 'no_car', 1: 'car'}

# Load video
video = cv2.VideoCapture("car_test.mp4")

# Load positions
with open('carposition.pkl', 'rb') as f:
    positionList = pickle.load(f)

# Width and height for bounding boxes
width = 130
height = 65

def checkingCarParking(img):
    imgCrops = []
    spaceCounter = 0
    for pos in positionList:
        if len(pos) == 4:  # If the position has 4 elements (bounding box coordinates)
            x, y, w, h = pos
        elif len(pos) == 2:  # If the position has 2 elements (top-left corner)
            x, y = pos
            w, h = width, height  # Use default width and height if only the top-left is given
        else:
            continue
        
        cropped_img = img[y:y+h, x:x+w]
        imgResized = cv2.resize(cropped_img, (48, 48))
        imgNormalized = imgResized / 255.0
        imgCrops.append(imgNormalized)
    
    imgCrops = np.array(imgCrops)
    predictions = model.predict(imgCrops)

    for i, pos in enumerate(positionList):
        if len(pos) == 4:  # Check again for 4-element bounding boxes
            x, y, w, h = pos
        elif len(pos) == 2:  # Check for 2-element top-left positions
            x, y = pos
            w, h = width, height
        else:
            continue
        
        intId = np.argmax(predictions[i])
        label = class_dictionary[intId]
        if label == 'no_car':
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
            textColor = (0, 0, 0)
        else:
            color = (0, 0, 255)
            thickness = 2
            textColor = (255, 255, 255)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        font_scale = 0.5
        text_thickness = 1
        
        textSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        textX = x
        textY = y + h - 5
        cv2.rectangle(image, (textX, textY - textSize[1] - 5), (textX + textSize[0] + 6, textY + 2), color, -1)
        cv2.putText(image, label, (textX + 3, textY - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, textColor, text_thickness)
    
    cv2.putText(image, f'Space Count: {spaceCounter}', (100, 50), (cv2.FONT_HERSHEY_SIMPLEX), 1, textColor, 2)

while True:
    if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
        video.set(cv2.CAP_PROP_FRAMES, 0)
    ret, image = video.read()
    image = cv2.resize(image, (1280, 720))
    if not ret:
        break
    checkingCarParking(image)
    cv2.imshow("Image", image)
    if cv2.waitKey(10) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
