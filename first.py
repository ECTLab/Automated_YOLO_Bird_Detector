import numpy as np
import argparse
import time
import cv2
import os
import keyboard

#The needed instructions for get the project running
#Open command prompt and go to the directory that this file is there
#Install the Numpy and CV2 libraries if needed.
#Run the program with this command "python first.py -d the directory  -y yolo-coco
#Change the confidence and threshold if needed too

ap = argparse.ArgumentParser()
ap.add_argument('-d','--directory',required=True,
	help="base path directory of images")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

#Adding the names files

labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

#Adding the weights

weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

directory = f"C:/Users/ehsan/Desktop/project/{args['directory']}"
run = True
tmp = 0

#Checking the addded picture if there is a bird in the picture

while run:
	time.sleep(2)
	count_files = len(os.listdir(directory))
	if count_files != 0:
		print(count_files)
		for img_path in os.listdir(directory):
			path = os.path.join(f'{directory}/{img_path}')
			print(path)
			image = cv2.imread(path)
			image = cv2.resize(image, (600,400))
			(H, W) = image.shape[:2]
			ln = net.getLayerNames()
			ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
			blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			net.setInput(blob)
			start = time.time()
			layerOutputs = net.forward(ln)
			end = time.time()
			print("[INFO] YOLO took {:.6f} seconds".format(end - start))


			boxes = []
			confidences = []
			classIDs = []
			
			#Localization of the detected bird

			for output in layerOutputs:
				for detection in output:
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]
					if confidence > args["confidence"]:
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
				args["threshold"])

			#Giving the alarm if a bird is detected
			
			if len(idxs) > 0:
				for i in idxs.flatten():
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])
					color = [int(c) for c in COLORS[classIDs[i]]]
					text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
					if LABELS[classIDs[i]] == 'bird':
						#cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
						#cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
						#	0.5, color, 2)
						alert = np.zeros((200,700,3)).astype(np.uint8)
						cv2.putText(alert, 'birds detected turn on the alarms', (0,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
						cv2.imshow('alert',alert)

					else:
						alert = np.zeros((200,700)).astype(np.uint8)
						cv2.putText(alert, 'no birds detected', (0,100), cv2.FONT_HERSHEY_COMPLEX, 1, (255,2,2), 1)
						cv2.imshow('alert',alert)
			else:
				alert = np.zeros((200,400,3)).astype(np.uint8)
				cv2.putText(alert, 'no birds detected', (0,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
				cv2.imshow('alert',alert)
			cv2.imshow("Image", image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			os.remove(path)
	#How to stop the program just type the "p" character
	elif count_files == 0:
		if keyboard.read_key() == "p":
			run = False
