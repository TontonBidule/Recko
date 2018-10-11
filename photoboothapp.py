# import the necessary packages
import tkinter as tki
from PIL import Image
from PIL import ImageTk

import cv2
import threading
import imutils
import datetime
import os
import time
import traceback
import shutil

# FaceAnalyzer
import dlib
import numpy as np
import cv2
from keras.models import load_model
import glob
from imutils import face_utils
import fr_utils
import time
import tensorflow as tf

class PhotoBoothApp:

    def __new__(cls, video_stream, outputPath):
        if outputPath[0:2] == './':
            return super(PhotoBoothApp, cls).__new__(cls)
        else:
            raise ValueError(' The output path must be in the current directory.')

    def __init__(self, video_stream, outputPath):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.vs = video_stream
        self.outputPath = outputPath

        if not os.path.isdir(self.outputPath):
            os.mkdir(self.outputPath)

        self.face_analyzer = FaceAnalyzer(self.outputPath, self)

        self.frame = None
        self.thread = None
        self.stopEvent = None

        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None

        self.buildTkInterface()
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

    def buildTkInterface(self):
        # create a button, that when pressed, will take the current
        # frame and save it to file

        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

        btn = tki.Button(self.root, text="Snapshot!",
                         command=self.takeSnapshot)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10,
                 pady=10)

        btn = tki.Button(self.root, text="Flush Database!",
                         command=self.flushDatabase)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10,
                 pady=10)

        tki.Label(self.root, text= 'Qui es tu ?').pack()

        self.name_text = tki.StringVar()
        self.name_widget = tki.Entry(self.root, textvariable = self.name_text)
        self.name_widget.pack()

        self.alert_text = tki.StringVar()
        self.alert_widget = tki.Label(self.root, textvariable = self.alert_text)

        self.alert_widget.pack()


        self.listbox = tki.Listbox(self.root)
        self.listbox.pack()

        faces = [os.path.splitext(filename)[0] for filename in os.listdir(self.outputPath)]
        [self.listbox.insert(tki.END,face) for face in faces]

    def verifSnapshot(self, new_filename):

        print(self.name_text)
        if not os.path.isdir(self.outputPath):
            return False

        if new_filename == ".jpg":
            self.alert_text.set("Tu as oublié le prénom ! >o<")
            self.alert_widget.config(fg="red")
            return False

        if not os.path.isfile("./people/" + new_filename):
            self.alert_text.set("Visage ajouté avec succès ! ^o^")
            self.alert_widget.config(fg="green")
            return True
        else:
            self.alert_text.set("Cette personne existe déja ! >o<")
            self.alert_widget.config(fg="red")
            return False

    def flushDatabase(self):

        if self.outputPath[0:2] == './':
            shutil.rmtree(self.outputPath)
            os.mkdir(self.outputPath)
            self.alert_text.set("Base de données vidée ! 'v'")
            self.alert_widget.config(fg="green")
            self.listbox.delete(0, tki.END)
            self.face_analyzer.reload_database()

    def videoLoop(self):

        try:
            while not self.stopEvent.is_set():

                (_, init_frame) = self.vs.read()

                self.frame = imutils.resize(init_frame, width=300)

                with graph.as_default():
                    frame_faces = self.face_analyzer.add_face_boxes()

                image = cv2.cvtColor(frame_faces, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                if self.panel is None:
                    self.panel = tki.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=10, pady=10)

                else:
                    self.panel.configure(image=image)
                    self.panel.image = image
            self.root.quit()

            print('[INFO] end of video thread.')

        except Exception as e:
            print(e)
            traceback.print_exc()
            print("[INFO] caught a RuntimeError")
            self.root.quit()

    def takeSnapshot(self):

        name = self.name_widget.get()
        filename = "{}.jpg".format(name)
        if self.verifSnapshot(filename):
            p = os.path.sep.join((self.outputPath, filename))
            # save the file
            face = self.face_analyzer.get_soloface_image()
            cv2.imwrite(p, face)
            print("[INFO] saved {}".format(filename))
            self.listbox.insert(tki.END, name)
            self.face_analyzer.reload_database()

    def onClose(self):

        print("[INFO] closing...")
        self.stopEvent.set()
        time.sleep(1)
        self.root.quit()

class FaceAnalyzer:

    def __init__(self, outputPath, photobooth_app):

        self.outputPath = outputPath
        self.database = {}

        self.FRmodel = load_model('face-rec_Google.h5')

        global graph
        graph = tf.get_default_graph()

        self.detector = dlib.get_frontal_face_detector()
        self.photobooth_app = photobooth_app
        self.reload_database()

    def reload_database(self):

        self.database = {}
        # load all the images of individuals to recognize into the database
        for photo_filename in glob.glob("%s/*" % (self.outputPath)):

            photo_object = cv2.imread(photo_filename)
            identity = os.path.splitext(os.path.basename(photo_filename))[0]
            self.database[identity] = fr_utils.img_path_to_encoding(photo_filename, self.FRmodel)

    def add_face_boxes(self):

        frame_copy = self.photobooth_app.frame.copy()

        faces = self.detector(frame_copy)
        x, y, w, h = 0, 0, 0, 0
        if len(faces) > 0:
            for face in faces:
                try:
                    (x, y, w, h) = face_utils.rect_to_bb(face)
                    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 255, 0), 2)

                    face_image = frame_copy[y:y + h, x:x + w].copy()
                    name, min_dist = self.recognize_face(face_image)

                    if min_dist < 0.15:
                        cv2.putText(frame_copy, "Face : " + name, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                        cv2.putText(frame_copy, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame_copy, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                except Exception as e:
                    print(e)

        return frame_copy

    def get_soloface_image(self):

        frame_copy = self.photobooth_app.frame
        faces = self.detector(frame_copy)

        if len(faces) == 0:
            self.alert_text.set("Aucun visage à l'horizon ! >o<")
            self.alert_widget.config(fg="red")
            return False

        if len(faces) == 1:
            try:
                face = faces[0]
                (x, y, w, h) = face_utils.rect_to_bb(face)
                face_image = frame_copy[y:y + h, x:x + w]
            except Exception as e:
                print(e)

            return face_image

        if len(faces) > 1:
            self.alert_text.set("Il y a plusieurs visages ! >o<")
            self.alert_widget.config(fg="red")
            return False

    def recognize_face(self, face_descriptor):
        encoding = fr_utils.img_to_encoding(face_descriptor, self.FRmodel)
        min_dist = 100
        identity = None

        # Loop over the database dictionary's names and encodings.
        for (name, db_enc) in self.database.items():

            # Compute L2 distance between the target "encoding" and the current "emb" from the database.
            dist = np.linalg.norm(db_enc - encoding)

            print('distance for %s is %s' % (name, dist))

            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
            if dist < min_dist:
                min_dist = dist
                identity = name

        return identity, min_dist
