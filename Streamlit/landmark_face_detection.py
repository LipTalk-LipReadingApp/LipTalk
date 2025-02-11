import cv2
import streamlit as st
import dlib 

def detect_faces_and_mouths(frame):

    hog_face_detector = dlib.get_frontal_face_detector()

    dlib_facelandmark = dlib.shape_predictor("../Streamlit/shape_predictor_68_face_landmarks.dat")

    # Initialize variables with default values
    y1, y2, x1, x2 = 0, 0, 0, 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)

    for face in faces:

        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        face_landmarks = dlib_facelandmark(gray, face)

        # Draw lines between landmarks
        for n in range(49, 68):
            if n == 49:
                start_point = (face_landmarks.part(n).x, face_landmarks.part(n).y)
            else:
                end_point = (face_landmarks.part(n).x, face_landmarks.part(n).y)
                cv2.line(frame, start_point, end_point, (0, 255, 255), 1)
                start_point = end_point  # Update start_point for the next iteration

        # Calculate bounding box for all landmarks
        x1 = min(face_landmarks.part(n).x for n in range(49, 68))
        y1 = min(face_landmarks.part(n).y for n in range(49, 68))
        x2 = max(face_landmarks.part(n).x for n in range(49, 68))
        y2 = max(face_landmarks.part(n).y for n in range(49, 68))

        cv2.rectangle(frame, (x1-12, y1-10), (x2+12, y2+10), (255, 0, 0), 2)

        
    return frame
