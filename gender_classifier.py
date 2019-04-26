import wget
import os
import tarfile
import numpy as np
from random import shuffle
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import os


class face_detector:
    def __init__(self, xml_path):
        self.face_cascade = cv2.CascadeClassifier(xml_path)

    def crop_face(self, img_path):
        """
        output : cv2.img object
        """
        image = cv2.imread(img_path)
        faces = self.face_cascade.detectMultiScale(image, 1.3, 5)

        if len(faces) > 0:
            x, y, h, w = faces[0]
            face = image[y:y+h, x:x+w]
        else:
            face = []

        return face
        


class gender_classifier:
    def __init__(self):
        self.root_path = os.getcwd()
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,0.5, 0.5), (0.5, 0.5, 0.5))])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.class_label = {
            0 : 'female',
            1 : 'male',
        }

        # load opencv face detector
        os.system("git clone https://github.com/opencv/opencv")
        self.face_detector = face_detector('opencv/data/haarcascades/haarcascade_frontalface_default.xml')

    def train_model(self, data, data_path='data'):
        #if data == 'wiki':
        #    url = 'https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar'
        #    data_path = os.path.join(self.root_path, data_path)
        #    tar_path = data_path + '/wiki_crop.tar'
        #    filename = wget.download(url, data_path)

        #    if not tarfile.is_tarfile(os.path.join(data_path, filename)):
        #        assert("download failed")
        #    
        #    tar = tarfile.open(tar_path)
        #    tar.extractall()
        #    tar.close()

        #    mat = scipy.io.loadmat(data_path + 'wiki_crop/wiki.mat')
	#    
	#    gender_label = mat[data_name][0, 0]["gender"][0]
        #    image_path = mat[data_name][0, 0]["full_path"][0]
        #    
        #    valid_idx = [i for i in range(len(gender_label)) if not np.isnan(gender_label[i])]
        #    face_cascade = cv2.CascadeClassifier(cv2_data_path + )
        #    
        #    croped_idx = []
        #    for i, idx in enumerate(valid_idx):
        #        path = data_name + '_crop/' + image_path[idx][0]
        #        image = cv2.imread(path)
        #        face = crop_face(image, 'opencv')
        #        
        #        if len(face) > 0:
        #            cv2.imwrite(path, face)
        #            croped_idx.append(idx)
        #        
        #        if i % 1000 == 0:
        #            print("done {}/{}".format(i, len(valid_idx)))
        #    
        #    shuffle(croped_idx)
        #    train_size = int(len(croped_idx) * 0.8)
        #    
        #    train_idx = croped_idx[:train_size]
        #    test_idx = croped_idx[train_size:]
        #    
        #    train_X = image_path[train_idx]
        #    train_y = gender_label[train_idx]
        #    
        #    test_X = image_path[test_idx]
        #    test_y = gender_label[test_idx]
        #    
        #    del mat

        pass
        
	
    def load_model(self, path):
        self.net = torch.load(path, map_location=lambda storage, loc: storage).to(self.device)

    def predict(self, img_path):
        face = self.face_detector.crop_face(img_path)
        img = Image.fromarray(face)
        img = self.transform(img).unsqueeze(0).to(self.device)

        outputs = self.net(img)
        _, predicted = torch.max(outputs, 1)

        return self.class_label[predicted.item()]
        

if __name__ == "__main__":
    gc = gender_classifier()
    gc.load_model('model/gender_cf_resnet.pt')
    print(gc.predict('06324.jpg'))

    #gc.train_model('wiki')
