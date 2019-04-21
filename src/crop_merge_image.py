import cv2
import sys
import os
import wand
from PIL import Image


class FaceCropper(object):

    def __init__(self, CASCADE_PATH):
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # real_img -> crop_img & masking_img method
    def generate(self, image_path, show_result):
        img = cv2.imread(image_path)
        if (img is None):
            print("Can't open image file")
            return 0

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img, 1.1, 2, minSize=(80, 80))
        if (faces is None):
            print('Failed to detect face')
            return 0

        facecnt = len(faces)
        print("Detected faces: %d" % facecnt)

        maskingimg = img

        for (x, y, w, h) in faces:
            print("x :", x, "y : ", y, "w :", w, "h :", h)
            faceimg = img[y:y + h, x:x + w]
            lastimg = cv2.resize(faceimg, (w, h))

            cv2.rectangle(maskingimg, (x, y), (x + w, y + h), (255, 255, 255), -1)

        cv2.imwrite("mask_face\\" + image_path.split("\\")[-1], maskingimg)
        cv2.imwrite("crop_face\\" + image_path.split("\\")[-1], lastimg)

        return faces

    #crop_image와 masked_image를 합성하는 함수
    def composite_face(self, faces, crop_image_path, masked_image_path, show_result):
        mask_img = cv2.imread(masked_image_path)
        if (mask_img is None):
            print("Can't open masking image file")
            return 0

        crop_img = cv2.imread(crop_image_path)
        if (crop_img is None):
            print("Can't open cropped image file")
            return 0
        rows, cols, channels = crop_img.shape

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if (faces is None):
            print('Failed to get information of face')
            return 0
        #rows, cols, channels = crop_img.shape


        for (x, y, w, h) in faces:
            #if rows < cols:
            resize_crop_img = cv2.resize(crop_img, (w, h))
            mask_img[y:y+h, x:x+w] = resize_crop_img

        cv2.imwrite("merge_face\\" + masked_image_path.split("\\")[-1], mask_img)





