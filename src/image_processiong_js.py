import crop_merge_image as cmi
import os
import numpy as np
import random
import cv2


if __name__ == '__main__':

    detector = cmi.FaceCropper("xml\\haarcascade_frontalface_default.xml")

    js_list = np.array([0,0,0,0])
    print(type(js_list))
    for file_name in os.listdir("csimg"):
        if file_name.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
            #print(file_name.split(".")[-1].lower())
            temp = detector.generate("csimg\\" + file_name, False)
            if np.all(np.array([0,0,0,0])== temp):
                continue
            else:
                js_list = np.vstack([js_list, temp])

    ##check
    print(len(js_list))
    for i in range(len(js_list)):
        print(js_list[i])


    print(type(js_list))
    print(js_list.shape)
    print(js_list.ndim)

    i=1
    #random composite code
    for file_name_mask in os.listdir("mask_face"):
        if file_name_mask.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
            while True:
                crop_filename = random.choice(os.listdir("C:\\Users\\USER\\Desktop\\croptest\\crop_face"))
                if crop_filename.split(".")[-2] != file_name_mask.split(".")[-1]:
                    break
            if crop_filename.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
                print("crop : " + crop_filename + " mask : " + file_name_mask)
                detector.composite_face(js_list[i], "crop_face\\" + crop_filename, "mask_face\\" + file_name_mask,False)
            i += 1
'''
    for file_name_mask in os.listdir("mask_face"):
        if file_name_mask.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
            #print(file_name.split(".")[-1].lower())
            
            for file_name_crop in os.listdir("crop_face"):
                if file_name_crop.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
                    print("crop : "+file_name_crop+ " mask : "+file_name_mask)
                    detector.composite_face(js_list[i],"crop_face\\" + file_name_crop,"mask_face\\" + file_name_mask, False)
            #img = cv2.imread("real_face\\" + file_name)
            i += 1
'''
