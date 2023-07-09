import cv2
import mediapipe as mp
import time
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.shortcuts import redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class GestureRecognizer:
    def __init__(self):
        self.gesture_functions = {
            "Fist": self.is_fist,
            "Index pointing": self.is_index_pointing,
            "Thumb up": self.is_thumb_up,
            "Ok Sign": self.is_ok_sign,
            "V Sign": self.is_v_sign,
        }

    def recognize_gesture(self, landmark_list):
        for gesture, condition_func in self.gesture_functions.items():
            if condition_func(landmark_list):
                return gesture
        return "Unknown"

    def is_fist(self, landmark_list):
        thumb_to_index, index_to_middle, middle_to_ring, ring_to_pinky = self.calculate_distances(landmark_list)
        return thumb_to_index < 0.1 and index_to_middle < 0.1 and middle_to_ring < 0.1 and ring_to_pinky < 0.1

    def is_index_pointing(self, landmark_list):
        thumb_to_index, index_to_middle, middle_to_ring, ring_to_pinky = self.calculate_distances(landmark_list)
        return thumb_to_index > 0.2 and index_to_middle > 0.2 and middle_to_ring < 0.1 and ring_to_pinky < 0.1

    def is_thumb_up(self, landmark_list):
        thumb_to_index, index_to_middle, middle_to_ring, ring_to_pinky = self.calculate_distances(landmark_list)
        return thumb_to_index > 0.2 and index_to_middle < 0.1 and middle_to_ring < 0.1 and ring_to_pinky < 0.1

    def is_ok_sign(self, landmark_list):
        thumb_to_index, index_to_middle, middle_to_ring, ring_to_pinky = self.calculate_distances(landmark_list)
        return thumb_to_index < 0.1 and index_to_middle > 0.2 and middle_to_ring < 0.1 and ring_to_pinky < 0.1

    def is_v_sign(self, landmark_list):
        thumb_to_index, index_to_middle, middle_to_ring, ring_to_pinky = self.calculate_distances(landmark_list)
        return (
            thumb_to_index > 0.2
            and index_to_middle > 0.2
            and middle_to_ring > 0.2
            and ring_to_pinky > 0.2
        )

    @staticmethod
    def calculate_distances(landmark_list):
        thumb_tip = landmark_list[4]
        index_finger_tip = landmark_list[8]
        middle_finger_tip = landmark_list[12]
        ring_finger_tip = landmark_list[16]
        pinky_finger_tip = landmark_list[20]

        thumb_to_index = calculate_distance(thumb_tip, index_finger_tip)
        index_to_middle = calculate_distance(index_finger_tip, middle_finger_tip)
        middle_to_ring = calculate_distance(middle_finger_tip, ring_finger_tip)
        ring_to_pinky = calculate_distance(ring_finger_tip, pinky_finger_tip)

        return thumb_to_index, index_to_middle, middle_to_ring, ring_to_pinky


@login_required
def index(request):
    return render(request, 'index.html')


def video_feed(request):
    response = StreamingHttpResponse(capture_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
    return response


def login_view(request):
    error_message = None
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            
            error_message = 'Invalid username or password'
            

    return render(request, 'login.html', {'error_message': error_message})


def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Your account has been created successfully. Please log in.')
            return redirect('login')
        else:
            print(form.errors) 
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})


# Calculate distance between two points
def calculate_distance(point1, point2):
    x1, y1, _ = point1
    x2, y2, _ = point2
    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return distance


# Function to capture video frames
def capture_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(3, 640)
    cap.set(4, 480)
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    gesture_recognizer = GestureRecognizer()

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1) 

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    mp_drawing.draw_landmarks(
                        frame_bgr,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

                    landmark_list = []
                    for landmark in hand_landmarks.landmark:
                        landmark_list.append((landmark.x, landmark.y, landmark.z))

                    gesture = gesture_recognizer.recognize_gesture(landmark_list)
                    hand_label = handedness.classification[0].label 

                    hand_label = handedness.classification[0].label

                    hand_open = "Open" if gesture != "Fist" else "Closed"

                    cTime = time.time()
                    fps_frame_count += 1
                    if (cTime - fps_start_time) > 1:
                        fps = fps_frame_count / (cTime - fps_start_time)
                        fps_frame_count = 0
                        fps_start_time = cTime

                    # Draw hand label and hand open/close status near the hand
                    hand_label_position = (int(hand_landmarks.landmark[0].x * frame_bgr.shape[1]),
                                           int(hand_landmarks.landmark[0].y * frame_bgr.shape[0]) - 10)
                    cv2.putText(frame_bgr, f'Hand: {hand_label}', hand_label_position, cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)
                    cv2.putText(frame_bgr, f'Status: {hand_open}', (hand_label_position[0], hand_label_position[1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.putText(frame_bgr, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame_bgr, f'Gesture: {gesture}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, jpeg = cv2.imencode('.jpg', frame_bgr)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
