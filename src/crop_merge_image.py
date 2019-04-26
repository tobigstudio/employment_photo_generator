import cv2
import sys
import os
import wand
from PIL import Image
from copy import deepcopy
import numpy as np
import operator

class FaceCropper(object):
    #클래스를 선언시 xml을 읽어들인다
    def __init__(self, CASCADE_PATH):
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # real_img -> crop_img & masking_img method
    def generate(self, image_path, show_result):
        img = cv2.imread(image_path)
        if (img is None):
            print("Can't open image file")
            return 0

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img, 1.15, 6, minSize=(70, 70))
        if (faces is None):
            print('Failed to detect face')
            return 0
        #detect된 face갯수를 facecnt에 저장
        facecnt = len(faces)
        print("Detected faces: %d" % facecnt)
        #검출된 얼굴이 없을 시 [0,0,0,0]을 반환
        if facecnt ==0 :
            return [0,0,0,0]
        #검출된 얼굴이 1개 이상일 경우
        #print(type(faces))

        #검출된 face중 width가 가장 큰 face의 index를 i로 설정
        i = np.where(faces == max(faces,  key=lambda x: x[2]))

#        i = faces.index(max(faces, key=lambda x: x[2]))


        print(image_path)

        #검출된 얼굴의 왼쪽아래의 x,y좌표와 width,height를 저장
        x = faces[i][0]
        y = faces[i][1]
        w = faces[i][2]
        h = faces[i][3]


        print("x :", x, "y : ", y, "w :", w, "h :", h)
        #crop된 이미지를 faceimg/lastimg로 저장
        faceimg = img

        faceimg = faceimg[y:y + h, x:x + w]
        lastimg = cv2.resize(faceimg, (w, h))
        maskingimg = img


#        if facecnt > 1:
#            maskingimg = deepcopy(img)
#        else :
#            maskingimg = img

        cv2.rectangle(maskingimg, (x, y), (x + w, y + h), (255, 255, 255), -1)

        #real_file_name= image_path.split("\\")[-1]

        cv2.imwrite("mask_face\\" + image_path.split("\\")[-1], maskingimg)
        cv2.imwrite("crop_face\\" + image_path.split("\\")[-1], lastimg)
        #numpy.ndarray형태의 face x,y,w,h값 출력
        print(type(faces[i]))
        return faces[i]


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

        x = faces[0]
        y = faces[1]
        w = faces[2]
        h = faces[3]

        print(x,y,w,h)

        #if rows < cols:
        resize_crop_img = cv2.resize(crop_img, (w, h))

        mask_img[y:y+h, x:x+w] = resize_crop_img

        cv2.imwrite("merge_face\\" + (masked_image_path.split("\\")[-1]).split(".")[-2]+"_"+crop_image_path.split("\\")[-1], mask_img)
