# importing libraries
import os
import skimage
import numpy as np
from numpy import expand_dims
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps

class ImageAugmentation:
    def __init__(self, image: str):
        self.image = load_img(image)
        self.gray_image = skimage.color.rgb2gray(self.image)
        
    def __init__(self, image: Image):
        self.image = image
        self.gray_image = skimage.color.rgb2gray(self.image)
        
    # Chuyển màu xám
    def ChangeColor(self):
        #plt.imshow(gray_image, cmap = 'gray')
        return self.gray_image
    
    # Bình thường hóa
    def Normalize(self):
        norm_image = (self.gray_image - np.min(self.gray_image)) / (np.max(self.gray_image) - np.min(self.gray_image))
        #plt.imshow(norm_image)
        return norm_image
    
    # Lật ngang
    def HorizontalFlip(self):
        # convert to numpy array
        dataImage  = img_to_array(self.image)

        # expand dimension to one sample
        imageNew  = expand_dims(dataImage, 0)
        
        imageDataGen = ImageDataGenerator(horizontal_flip=True)
        
        iterator = imageDataGen.flow(imageNew, batch_size=1, save_to_dir ='Yolov3-custom-transfer-learning/trainning_images_augmetation_horizotal/', 
                          save_prefix ='image', save_format ='jpg', shuffle=False)
        
        return iterator.next()[0].astype('uint8')
    
    # Lật dọc
    def VerticalFlip(self):
        # convert to numpy array
        dataImage  = img_to_array(self.image)

        # expand dimension to one sample
        imageNew  = expand_dims(dataImage, 0)
        
        imageDataGen = ImageDataGenerator(vertical_flip=True)
        
        iterator = imageDataGen.flow(imageNew, batch_size=1, save_to_dir ='Yolov3-custom-transfer-learning/trainning_images_augmetation_vertical/', 
                          save_prefix ='image', save_format ='jpg', shuffle=False)
        
        return iterator.next()[0].astype('uint8')
        
    # Xoay
    def Rotate(self):
        # convert to numpy array
        dataImage  = img_to_array(self.image)

        # expand dimension to one sample
        imageNew  = expand_dims(dataImage, 0)
        
        imageDataGen = ImageDataGenerator(rotation_range=45, fill_mode='nearest')
        
        i = 0
        for batch in imageDataGen.flow(imageNew, batch_size = 1,
                          save_to_dir ='Yolov3-custom-transfer-learning/trainning_images_augmetation_rotate/', 
                          save_prefix ='image', save_format ='jpg'):
            i += 1
            if i > 3:
                break
       
    # Xoay
    def Rotate(self, degree):
        # convert to numpy array
        dataImage  = img_to_array(self.image)

        # expand dimension to one sample
        imageNew  = expand_dims(dataImage, 0)
        
        imageDataGen = ImageDataGenerator(rotation_range=degree, fill_mode='nearest')
        
        i = 0
        for batch in imageDataGen.flow(imageNew, batch_size = 1,
                          save_to_dir ='Yolov3-custom-transfer-learning/trainning_images_augmetation_rotate90/', 
                          save_prefix ='image', save_format ='jpg', shuffle=False):
            i += 1
            if i > 2 :
                break   
            
    # Thay đổi độ sáng
    def ChangeLight(self):
        # convert to numpy array
        dataImage  = img_to_array(self.image)

        # expand dimension to one sample
        imageNew  = expand_dims(dataImage, 0)
        
        imageDataGen = ImageDataGenerator(brightness_range=[0.1,2.0])
        
        i = 0
        for batch in imageDataGen.flow(imageNew, batch_size = 1,
                          save_to_dir ='Yolov3-custom-transfer-learning/trainning_images_augmetation_light/', 
                          save_prefix ='image', save_format ='jpg'):
            i += 1
            if i > 3:
                break
    
    def ChangeImageAugemtation(self):
        # convert to numpy array
        dataImage  = img_to_array(self.image)

        # expand dimension to one sample
        imageNew  = expand_dims(dataImage, 0)
        
        imageDataGen = ImageDataGenerator(rotation_range=90, vertical_flip=True, horizontal_flip=True)
        
        i = 0
        for batch in imageDataGen.flow(imageNew, batch_size = 1,
                          save_to_dir ='Yolov3-custom-transfer-learning/trainning_images_augmetation/', 
                          save_prefix ='image', save_format ='jpg', shuffle=False):
            i += 1
            if i > 2 :
                break  
    
folderDirec = "Yolov3-custom-transfer-learning/trainning_images/"
directory = os.fsencode(folderDirec)
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"): 
        im = Image.open(folderDirec + filename)
        old_size = im.size  # old_size[0] is in (width, height) format
        h, w = im.size
        desired_size = max(h,w)
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        im = im.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))
        
        image = ImageAugmentation(new_im)
        image.ChangeImageAugemtation()