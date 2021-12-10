
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import torch
import glob

_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

class GTOS_mobile_single_data(Dataset):

    def __init__(self, texture_dir, kind = 'train', split_board=0.9, image_size = 384, img_transform = None):  # numset: 0~5 #paper:256
        self.texture_dir = texture_dir
        self.img_transform = img_transform
        self.files = []  # empty list
        self.targets = [] #labels
        self.paths = []
        self.split_board = split_board
        #pdb.set_trace()
        imgset_dir = os.path.join(self.texture_dir)

        if kind == 'train':  # train
            #Get training file
            sample_dir = os.path.join(imgset_dir,'train')
            class_names = sorted(os.listdir(sample_dir))
            label = 0
            #Loop through data frame and get each image

            for img_folder in class_names:
                #Set class label 
                #Select folder and remove class number/space in name
                temp_img_folder = os.path.join(sample_dir, img_folder)
                # temp_img_number = len(glob.glob(pathname=temp_img_folder+'/{}*.jpg'.format(str(image_size))))  # 获取当前文件夹下个数
                # count = 0
                for image in os.listdir(temp_img_folder):

                    #Check for correct size image
                    if not image.startswith(str(image_size)):
                        continue
                    if(image=='Thumbs.db'):
                        print('Thumb image')
                    else:
                        # count = count + 1
                        img_file = os.path.join(temp_img_folder,image)
                        self.files.append({  # appends the images
                                "img": img_file,
                                "label": label
                            })
                        self.paths.append(img_file)
                        self.targets.append(label)
                        # if count > int(self.split_board * temp_img_number):
                        #     break
                label += 1

        elif kind=='test':  # test
            sample_dir = os.path.join(imgset_dir,'test')#'test')
            class_names = sorted(os.listdir(sample_dir))
            label = 0
            #Loop through data frame and get each image
            for img_folder in class_names:
                #Set class label 
                #Select folder and remove class number/space in name
                temp_img_folder = os.path.join(sample_dir,img_folder)
                for image in os.listdir(temp_img_folder):
                    if(image=='Thumbs.db'):
                        print('Thumb image') 
                    else:
                        img_file = os.path.join(temp_img_folder,image)
                        self.files.append({  # appends the images
                                "img": img_file,
                                "label": label
                            })
                        self.paths.append(img_file)
                        self.targets.append(label)
                label +=1
        self.class_names = class_names
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')

        label_file = datafiles["label"]
        label = torch.tensor(label_file)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, label

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

# if __name__ == '__main__':
#     path = 'C:/Users/Zicong/Desktop/resnet texture/Histogram_Layer-master/Datasets/gtos-mobile/'
#     train = GTOS_mobile_single_data(path, kind='train',split_board=0.8)
#     print(len(train))
#     valtrain = GTOS_mobile_single_data(path, kind='val_in_train',split_board=0.8)
#     print(len(valtrain))

# #     test = GTOS_mobile_single_data(path, train=False)