import cv2
import dlib
import pyautogui
import mediapipe as mp
from scipy.spatial import distance
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
flag = 0
def calculate_ear(eye):
	a = distance.euclidean(eye[0], eye[3])
	b = distance.euclidean(eye[1], eye[5])
	c = distance.euclidean(eye[2], eye[4])
	eye_aspect_ratio = (b+c)/(2*a)
	return eye_aspect_ratio


def calculate_mar(mou):
	a = distance.euclidean(mou[0], mou[4])
	b = distance.euclidean(mou[1], mou[7])
	c = distance.euclidean(mou[2], mou[6])
	d = distance.euclidean(mou[3], mou[5])
	mouth_aspect_ratio = (b + c + d) / (2 * a)
	return mouth_aspect_ratio


cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
	_, frame = cap.read()
	frame = cv2.flip(frame, 1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = hog_face_detector(gray)
	for face in faces:
		face_landmarks = dlib_facelandmark(gray, face)
		lefteye = []
		righteye = []
		mouth = []
		landmarks = []
		for n in range(36,42):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			lefteye.append((x,y))
			next_pt = n+1
			if n == 41:
				next_pt = 36
			x2 = face_landmarks.part(next_pt).x
			y2 = face_landmarks.part(next_pt).y
			cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

		for n in range(42,48):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			landmarks.append(face_landmarks.part(n))
			righteye.append((x,y))
			next_pt = n+1
			if n == 47:
				next_pt = 42
			x2 = face_landmarks.part(next_pt).x
			y2 = face_landmarks.part(next_pt).y
			cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

		for n in range(60, 68):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			mouth.append((x, y))
			next_pt = n+1
			if n == 67:
				next_pt = 60
			x2 = face_landmarks.part(next_pt).x
			y2 = face_landmarks.part(next_pt).y
			cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

		left_ear = calculate_ear(lefteye)
		right_ear = calculate_ear(righteye)
		mouth_mar = calculate_mar(mouth)
		total_ear = (left_ear + right_ear)/2
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		output = face_mesh.process(rgb_frame)
		landmark_points = output.multi_face_landmarks
		frame_h, frame_w, _ = frame.shape
		if mouth_mar > 0.5:
			if flag == 0:
				flag=1
			elif flag == 1:
				flag = 0
			else:
				continue
			pyautogui.sleep(1)
		if total_ear < 0.15:
			if flag == 0:
				flag = 2
			elif flag == 2:
				flag = 0
			else:
				continue
			pyautogui.sleep(1)
		if flag == 1:
			if landmark_points:
				landmarks = landmark_points[0].landmark
				for id, landmark in enumerate(landmarks[474:478]):
					x = int(landmark.x * frame_w)
					y = int(landmark.y * frame_h)
					cv2.circle(frame, (x, y), 3, (0, 255, 0))
					if id == 1:
						screen_x = screen_w * landmark.x
						screen_y = screen_h * landmark.y
						pyautogui.moveTo(screen_x, screen_y)
					print(left_ear,right_ear)	
					if right_ear < 0.15:
						pyautogui.rightClick()
						pyautogui.sleep(1)
					if left_ear < 0.16:
						pyautogui.click()
						pyautogui.sleep(1)
		if flag == 2:
			m = 265-face_landmarks.part(33).y
			pyautogui.scroll(m)

	cv2.imshow("mouse control", frame)
	key = cv2.waitKey(1)
	if key == 27:
		break
cap.release()
cv2.destroyAllWindows()




























































