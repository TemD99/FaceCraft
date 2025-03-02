import cv2
import os
import json
from deepface import DeepFace
from glasses_detector import AnyglassesClassifier

img_folder = 'database_128/'

glasses_classifier = AnyglassesClassifier(base_model="large", pretrained=False)

labels = []
img_count = 0
noface_count = 0
noface = []

for subdir in os.listdir(img_folder):
    subdir_path = os.path.join(img_folder, subdir)
    if os.path.isdir(subdir_path):
        for filename in os.listdir(subdir_path):
            if filename.endswith(".png"):
                image_path = os.path.join(subdir_path, filename)
                image = cv2.imread(image_path)
                img_count += 1
                print(img_count)

                try:
                    analysis = DeepFace.analyze(image, actions=['age', 'gender', 'emotion', 'race'], silent=True)
                    analysis_result = analysis[0]
                    hasGlasses = glasses_classifier.predict(image)
                    label = f"{analysis_result['dominant_race']} {analysis_result['age']} years old {analysis_result['dominant_gender']}"
                    if hasGlasses: label += " with glasses"
                    label += f", appears {analysis_result['dominant_emotion']}"
                    labels.append([image_path, label])
                except Exception as e:
                    #print(f"Error processing {image_path}: {str(e)}")
                    labels.append([image_path, "n/a"])
                    noface_count += 1
                    noface.append(image_path)

with open("image_descriptions.json", "w") as json_file:
    json.dump({"labels": labels}, json_file, indent=4)

print("Descriptions saved to image_descriptions.json.")
print(f"Times unable to read face:{noface_count}/{img_count}")
