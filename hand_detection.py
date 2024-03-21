import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

image_path = '/path/to/image.png'
detection_threshold = 0.5

base_options = python.BaseOptions(model_asset_path='asserts/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
										num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

image = mp.Image.create_from_file(image_path)
detection_result = detector.detect(image)

if len(detection_result.hand_landmarks) > 0:
	thredhold = detection_result.handedness[0][0].score

	if thredhold > detection_threshold:
		print(f'Found hands in {image_path}!')
	