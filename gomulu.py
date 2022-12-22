from mfrc522 import SimpleMFRC522
import cv2
import numpy as np
import glob
import dlib
import os
import sys
import time
from time import sleep
import RPi.GPIO as GPIO
GPIO.cleanup()
predictor_path = "./shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

RED_PIN = 36
BLUE_PIN = 38
GREEN_PIN = 40
BUTTON_PIN = 11


def capture_face():
    camera = cv2.VideoCapture(0)
    camera.set(3, 320)
    camera.set(4, 320)
    time.sleep(1.5)
    return_value, img = camera.read()
    del(camera)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def dlibVect_to_numpyNDArray(vector):
    array = np.zeros(shape=128)
    for i in range(0, len(vector)):
        array[i] = vector[i]
    return array


def compare_faces(arr_1, arr_2):
    arr_1 = dlibVect_to_numpyNDArray(arr_1)
    arr_2 = dlibVect_to_numpyNDArray(arr_2)
    return 1 - np.linalg.norm(arr_1 - arr_2)


face_descriptors = {}

GPIO.setmode(GPIO.BOARD)

# set gpio 3 as pull down input
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# set 36, 38, 40 as output
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(BLUE_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)
# Set 36, 38, 40 to high, which means no light
GPIO.output(RED_PIN, 1)  # red
GPIO.output(BLUE_PIN, 1)  # blue
GPIO.output(GREEN_PIN, 1)  # green


def led_blue():
    GPIO.output(RED_PIN, 1)
    GPIO.output(BLUE_PIN, 0)
    GPIO.output(GREEN_PIN, 1)


def led_red():
    GPIO.output(RED_PIN, 0)
    GPIO.output(BLUE_PIN, 1)
    GPIO.output(GREEN_PIN, 1)


def led_green():
    GPIO.output(RED_PIN, 1)
    GPIO.output(BLUE_PIN, 1)
    GPIO.output(GREEN_PIN, 0)


def led_off():
    GPIO.output(RED_PIN, 1)
    GPIO.output(BLUE_PIN, 1)
    GPIO.output(GREEN_PIN, 1)


def error():
    led_red()
    sleep(0.2)
    led_off()
    sleep(0.2)
    led_red()
    sleep(0.2)
    led_off()


def registered():
    led_green()
    sleep(0.2)
    led_off()
    sleep(0.2)
    led_green()
    sleep(0.2)
    led_off()


def led_verified():
    led_green()
    sleep(1)
    led_off()


def led_denied():
    led_red()
    sleep(1)
    led_off()


try:
    while True:
        reader = SimpleMFRC522()
        read_result = reader.read_id_no_block()
        while read_result == None:
            time.sleep(0.2)
            reader = SimpleMFRC522()
            read_result = reader.read_id_no_block()
            print(read_result)
        print("Read card: {}".format(read_result))
        led_blue()
        if GPIO.wait_for_edge(BUTTON_PIN, GPIO.FALLING):
            # Read card and compare face.
            # wait for event on pin 5
            img = capture_face()
            dets = detector(img, 1)
            if len(dets) == 0:
                print("No faces detected!")
                error()
                continue
            for k, d in enumerate(dets):
                shape = sp(img, d)
                face_chip = dlib.get_face_chip(img, shape)
                face_descriptor = facerec.compute_face_descriptor(face_chip)
                if read_result in face_descriptors:
                    print("Comparing faces")
                    match = compare_faces(
                        face_descriptor, face_descriptors[read_result])
                    if match > 0.57:
                        print("Face matches!")
                        led_verified()
                    else:
                        print("Face doesn't match")
                        led_denied()
                    print(match)
                else:
                    print("Face not registered!")
                    face_descriptors[read_result] = face_descriptor
                    print("Face registered!")
                    registered()
                break
            time.sleep(0.3)
            continue
except:
    # print error
    print("Exiting...")
    GPIO.cleanup()
