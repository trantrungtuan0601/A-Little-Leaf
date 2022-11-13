# importing libraries
import os
import skimage
import numpy as np
from numpy import expand_dims
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import ImageDataGenerator

class ImageAugmentation:
    def __init__(self, image):
        self.image = load_img(image)
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
                          save_prefix ='image', save_format ='jpeg', shuffle=False)
        
        return iterator.next()[0].astype('uint8')
    
    # Lật dọc
    def VerticalFlip(self):
        # convert to numpy array
        dataImage  = img_to_array(self.image)

        # expand dimension to one sample
        imageNew  = expand_dims(dataImage, 0)
        
        imageDataGen = ImageDataGenerator(vertical_flip=True)
        
        iterator = imageDataGen.flow(imageNew, batch_size=1, save_to_dir ='Yolov3-custom-transfer-learning/trainning_images_augmetation_vertical/', 
                          save_prefix ='image', save_format ='jpeg', shuffle=False)
        
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
                          save_prefix ='image', save_format ='jpeg'):
            i += 1
            if i > 3:
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
                          save_prefix ='image', save_format ='jpeg'):
            i += 1
            if i > 3:
                break
    
folderDirec = "Yolov3-custom-transfer-learning/trainning_images/"
directory = os.fsencode(folderDirec)
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"): 
        image = ImageAugmentation(folderDirec + filename)
        image.VerticalFlip()
        #image.HorizontalFlip()
        image.Rotate()
        #image.ChangeLight()