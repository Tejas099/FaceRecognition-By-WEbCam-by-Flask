import cv2
# import math
# from sklearn import neighbors
# import os
# import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
# from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
# from datetime import datetime
# face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6
increment=0
knn_clf=None
model_path="trained_knn_model.clf"
distance_threshold=0.5
if knn_clf is None:
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        process_this_frame = 4
        # success, image = self.video.read()
        ret, frame =self.video.read()  
        if ret:
            # Different resizing options can be chosen based on desired program runtime.
            # Image resizing for more stable streaming
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            # img=cv2.resize(frame,None,fx=0.6,fy=0.6,interpolation=cv2.INTER_AREA)
            process_this_frame = process_this_frame + 1
            if process_this_frame % 5 == 0:
                X_face_locations = face_recognition.face_locations(img)
                if len(X_face_locations) == 0:
                    predictions= []
                else:
                    faces_encodings = face_recognition.face_encodings(img, known_face_locations=X_face_locations)

                    # Use the KNN model to find the best matches for the test face
                    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
                    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
                    predictions=[(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
                # predictions = predict(img, model_path="trained_knn_model.clf")
            pil_image = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_image)
            for name, (top, right, bottom, left) in predictions:
                # enlarge the predictions for the full sized image.
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                # Draw a box around the face using the Pillow module
                draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

                # There's a bug in Pillow where it blows up with non-UTF-8 text
                # when using the default bitmap font
                name = name.encode("UTF-8")

                # Draw a label with a name below the face
                text_width, text_height = draw.textsize(name)
                draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
                draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
            del draw
            opencvimage = np.array(pil_image)
            frame=opencvimage
        
        
        # for name,(x,y,w,h) in predictions:
        #     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #     break
        success, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()



