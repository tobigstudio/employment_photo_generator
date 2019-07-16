import wget
import os
import tarfile
import numpy as np
from random import shuffle, sample
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import os
from scipy import io
from tqdm import tqdm

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
        
class ImageDataset(data.Dataset):
    def __init__(self, root, image_paths, labels, transform=None):
        self.root = root
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = int(self.labels[idx])

        img_fullpath = self.root + '/' + img_path
        image = Image.open(img_fullpath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)

        
        
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

    def set_dataloader(self, data, crop_face=True, batch_size=16, num_workers=4):
        self.batch_size = batch_size
        
        if data == 'wiki' or data == 'imdb':
            if not os.path.isdir(os.path.join(self.root_path,'{}_crop'.format(data))):
                url = 'https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/{}_crop.tar'.format(data)
                data_path = os.path.join(self.root_path, data)
                filename = wget.download(url)
                tar_path = filename
                print("download end, {}".format(tar_path))

                if not tarfile.is_tarfile(tar_path):
                    assert("download failed")

                with tarfile.open(tar_path) as tar:
                    tar.extractall()
            
            mat = io.loadmat('{}_crop/{}.mat'.format(data, data))
            gender_label = mat[data][0, 0]["gender"][0]
            image_path = mat[data][0, 0]["full_path"][0]
            
            dataset = [(os.path.join('{}_crop'.format(data), p[0]), g) for p, g in zip(image_path, gender_label)]
            dataset = [d for d in dataset if not np.isnan(d[1])]
        
        else:
            """
                image_directory structure :
                    data_dir
                    |-- label0
                    |---- pic0
                    |---- pic1
                    |---- pic2
                    |-- label1
                    |---- pic3
                    |---- pic4
                    |---- pic5
            """
            data_dir = os.path.join(gc.root_path, data)
            labels = [folder_name for folder_name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder_name))]
            self.class_label = {}

            dataset = []

            for i, label in enumerate(labels):
                self.class_label[i] = label
                label_dir = os.path.join(data_dir, label)
                images = os.listdir(label_dir)
                    
                with tqdm(total=len(images), desc='load_images ' + label + '\t'*2) as progressbar:
                    for img_filename in images:
                        img_path = os.path.join(data, os.path.join(label, img_filename))
                        tmp_data = (img_path, i)
                        dataset.append(tmp_data)
                        progressbar.update()
                    
        if crop_face:
            def crop_face_subprocess(pool):
                sub_valid_data = []
                
                with tqdm(total=len(pool), desc='crop_face') as progressbar:
                    for data in pool:
                        path, _ = data
                        face = self.face_detector.crop_face(path)

                        if len(face) > 0:
                            cv2.imwrite(path, face)
                            sub_valid_data.append(data)
                            
                        progressbar.update()

                return sub_valid_data
            
            dataset = crop_face_subprocess(dataset)

        shuffle(dataset)
        train_size = int(len(dataset) * 0.8)

        train_X = [d[0] for d in dataset[:train_size]]
        train_y = [d[1] for d in dataset[:train_size]]

        test_X = [d[0] for d in dataset[train_size:]]
        test_y = [d[1] for d in dataset[train_size:]]

        trainset = ImageDataset(self.root_path, train_X, train_y, self.transform)
        testset = ImageDataset(self.root_path, test_X, test_y, self.transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    def load_model(self, path):
        self.net = torch.load(path, map_location=lambda storage, loc: storage).to(self.device)
        
    def save_model(self, path):
        torch.save(self.net, path)

    def train_model(self,
                    epoch=5,
                    init_model=True,
                    opt_lr=0.001,
                    opt_momentum=0.9,
                    silence=False):
        if init_model:
            self.net = models.resnet18(num_classes=len(self.class_label))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), 
                              lr=opt_lr, 
                              momentum=opt_momentum)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)

        n_epoch = epoch
        
        if not silence:
            print('''
            train info
                - epoch         : {},
                - batch_size    : {},
                - model         : {},
                - loss function : {},
                - optimizer     : {},
            '''.format(n_epoch, self.batch_size, 'resnet18', 'CrossEntropy', 'SGD'))

        for epoch in range(n_epoch):
            self.net.train()
            running_loss = 0.0
            print("epoch : {}".format(epoch))
            for i, data in enumerate(self.trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                if inputs.shape[0] is not self.batch_size:
                    continue
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if not silence and i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

            total = 0
            correct = 0

            if not silence:
                self.net.eval()
                with torch.no_grad():
                    for data in self.testloader:
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = self.net(inputs)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                    print('Accuracy : %0.2f%%' % (correct * 1.0 / total * 100))
        
    def predict(self, img_path, crop_face=True):
        self.net.eval()
        if crop_face:
            face = self.face_detector.crop_face(img_path)
            if len(face) == 0:
                return 'unknown'
            img = Image.fromarray(face)
        else:
            img = Image.open(img_path).convert('RGB')
        img = self.transform(img).unsqueeze(0).to(self.device)

        outputs = self.net(img)
        _, predicted = torch.max(outputs, 1)

        return self.class_label[predicted.item()]
        

if __name__ == "__main__":
    gc = gender_classifier()
    gc.set_dataloader('wiki', crop_face=True)
    gc.train_model(epoch=100, init_model=True, silence=False)

    import shutil

    data_dir = 'test'
    output_dir = 'gender'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for file_name in os.listdir(data_dir):
        if file_name.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
            path = os.path.join(data_dir, file_name)
            gender = gc.predict(path, crop_face=True)
            label_dir = os.path.join(output_dir, gender)
            if not os.path.isdir(label_dir):
                os.mkdir(label_dir)
            shutil.move(path, os.path.join(label_dir, file_name))