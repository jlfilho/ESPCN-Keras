import sys  
import os
import gc
import math
import numpy as np
import cv2
import glob
import imageio
from PIL import Image
from random import choice
from keras.utils import Sequence
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Lambda, Input
from keras.models import Model
import tensorflow as tf
from subprocess import Popen, PIPE
from timeit import default_timer as timer
from losses import psnr2 as psnr


class DataLoader(Sequence):
    def __init__(self, datapath, batch_size, height_hr, width_hr, scale, crops_per_image, media_type,channels=3,colorspace='RGB'):
        """        
        :param string datapath: filepath to training images
        :param int height_hr: Height of high-resolution images
        :param int width_hr: Width of high-resolution images
        :param int height_hr: Height of low-resolution images
        :param int width_hr: Width of low-resolution images
        :param int scale: Upscaling factor
        """

        # Store the datapath
        self.datapath = datapath
        self.batch_size = batch_size
        self.height_hr = height_hr
        self.height_lr = int(height_hr / scale)
        self.width_hr = width_hr
        self.width_lr = int(width_hr / scale)
        self.channels = channels
        self.colorspace = colorspace
        self.scale = scale
        self.crops_per_image = crops_per_image
        self.media_type  = media_type
        self.time_step=1
        self.total_imgs = None
        
        # Options for resizing
        self.options = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
        
        # Check data source
        self.img_paths = []

        if os.path.isdir(self.datapath):
            self.get_paths()
    
    def get_paths(self):
        for dirpath, _, filenames in os.walk(self.datapath):
            for filename in [f for f in filenames if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg','mp4','264','webm','wma'])]:
                self.img_paths.append(os.path.join(dirpath, filename))
        self.total_imgs = len(self.img_paths)
        print(">> Found {} images in dataset".format(self.total_imgs))
    
    def random_crop(self, img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y+dy), x:(x+dx), :]

    def fix_crop(self, img, dy, dx, y, x):
        return img[y:(y+dy), x:(x+dx), :]

    @staticmethod
    def scale_lr_imgs(imgs):
        """Scale low-res images prior to passing to SRGAN"""
        return imgs / 255.
    
    @staticmethod
    def unscale_lr_imgs(imgs):
        """Un-Scale low-res images"""
        return imgs * 255
    
    @staticmethod
    def scale_hr_imgs(imgs):
        """Scale high-res images prior to passing to SRGAN"""
        #return imgs / 127.5 - 1
        return imgs / 255.
    
    @staticmethod
    def unscale_hr_imgs(imgs):
        """Un-Scale high-res images"""
        #return (imgs + 1.) * 127.5
        return imgs * 255
    
    def count_frames_manual(self,cap):
        count=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if(ret):
                count +=1
            else:
                break
        return count
    
    def count_frames(self,cap):
        '''Count total frames in video'''
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            total = self.count_frames_manual(cap)
        return total


    def get_random_frames(self,n_fms=1,cap=None,time_step=1):
        """Get random number of video frames"""
        np.random.seed(int(1000000*(timer()%1)))
        t_frames = self.count_frames(cap)
        if(time_step==1):
            choiced_frames = np.random.randint(t_frames, size=n_fms)
        else:
            choiced_frames = np.random.randint(t_frames-time_step, size=n_fms)
        return choiced_frames
    
    
    @staticmethod
    def load_img(path,colorspace='YCbCr'):
        img = Image.open(path)
        if (colorspace ==  'YCbCr' and img.mode != 'YCbCr'):
            img = img.convert('YCbCr') 
        if (colorspace ==  'RGB' and img.mode != 'RGB'):
            img = img.convert('RGB')     
        return np.array(img)


    def load_frame(self,videopath,time_step=1,colorspace='YCbCr'):
        """Get n_imgs random frames from the video"""
        cap = cv2.VideoCapture(videopath)
        if(not cap.isOpened()):
            print("Error to open video: ",videopath)
            return -1 
        choiced_frame=self.get_random_frames(1,cap,time_step)
        cap.set(1,choiced_frame)
        frames = []
        for i in list(range(time_step)):
            ret, frame = cap.read()
            if not ret:
                print(">> Erro to access frames.")
                return -1
            if colorspace=='YCbCr':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:,:,:self.channel]
                frames.append(frame)
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
        cap.release()
        return np.array(frames)    


    def resize(self,shape,scale,image):
    
        # 3 channel images of arbitrary shape
        inp = Input(shape=shape)
        try:
            out = Lambda(lambda image: K.tf.image.resize_bicubic(image, (shape[0]//scale, shape[1]//scale)))(inp)
        except :
            # if you have older version of tensorflow
            out = Lambda(lambda image: K.tf.image.resize_bicubic(image, shape[0]//scale, shape[1]//scale))(inp)
        model = Model(inputs=inp, outputs=out)
        out = model.predict(image[np.newaxis, ...])
        return np.squeeze(out, axis=0).astype(np.uint8) 
   
    
    def __len__(self):
        return int(self.total_imgs / float(self.batch_size))
    
    def __getitem__(self, idx):
        return self.load_batch(idx=idx)        



    def load_batch(self,idx=0, img_paths=None, training=True, bicubic=False):
        """ Loads a batch of images or video"""
        if(self.media_type=='i'):
            #print("1. Media type image")
            imgs_lr, imgs_hr = self.load_batch_image(idx, img_paths=img_paths, training=training,bicubic=False)
        elif(self.media_type=='v' and os.path.isdir(self.datapath)):
            #print("2. Media type video folder")
            imgs_lr, imgs_hr = self.load_batch_video(idx, img_paths=None, training=training, bicubic=False)
        elif(self.media_type=='v' and not os.path.isdir(self.datapath)):
            #print("3. Media type single video")
            imgs_lr, imgs_hr = self.load_batch_video(idx, img_paths=self.datapath, training=training, bicubic=False)
        else:
            print("Error: type of media fail")
        return imgs_lr, imgs_hr


    def load_batch_video(self, idx=0, img_paths=None, training=True, bicubic=False):
        """Loads a batch of frames from videos folder""" 
        # Starting index to look in
        cur_idx = 0
        if not img_paths:
            cur_idx = idx*self.batch_size            
        #print('cur_idx:',cur_idx)
            
        # Scale and pre-process images
        imgs_hr, imgs_lr = [], []
        while True:

            # Check if done with batch
            if img_paths is None:
                if cur_idx >= self.total_imgs:
                    cur_idx = 0
                if len(imgs_hr) >= self.batch_size:
                    break
            if img_paths is not None and len(imgs_hr) == len(img_paths):
                break            
            
            try: 
                # Load image
                img_hr = None
                if img_paths:
                    #print('1. video: ',img_paths[cur_idx])
                    img_hr = self.load_frame(img_paths[cur_idx])
                    #print(img_hr.shape)
                else:
                    #print('2. video: ',self.img_paths[cur_idx])
                    img_hr = self.load_frame(self.img_paths[cur_idx])
                    #print(img_hr.shape)

                # Create HR images to go through
                img_crops = []
                if training:
                    for i in range(self.crops_per_image):
                        #print(idx, cur_idx, "Loading crop: ", i)
                        img_crops.append(self.random_crop(img_hr, (self.height_hr, self.width_hr)))
                else:
                    img_crops = [img_hr]

                # Downscale the HR images and save
                for img_hr in img_crops:

                    # TODO: Refactor this so it does not occur multiple times
                    if img_paths is None:
                        if cur_idx >= self.total_imgs:
                            cur_idx = 0
                        if len(imgs_hr) >= self.batch_size:
                            break
                    if img_paths is not None and len(imgs_hr) == len(img_paths):
                        break   

                    # For LR, do bicubic downsampling
                    method = Image.BICUBIC if bicubic else choice(self.options)
                    lr_shape = (int(img_hr.shape[1]/self.scale), int(img_hr.shape[0]/self.scale))           
                    img_lr = Image.fromarray(img_hr.astype(np.uint8))
                    img_lr = np.array(img_lr.resize(lr_shape, method))


                    # Scale color values
                    img_hr = self.scale_hr_imgs(img_hr)
                    img_lr = self.scale_lr_imgs(img_lr)

                    # Store images
                    imgs_hr.append(img_hr)
                    imgs_lr.append(img_lr)
                
            except Exception as e:
                print(e)
                pass
            finally:
                cur_idx += 1

        # Convert to numpy arrays when we are training 
        # Note: all are cropped to same size, which is not the case when not training
        if training:
            imgs_hr = np.array(imgs_hr)
            imgs_lr = np.array(imgs_lr)

        # Return image batch
        return imgs_lr, imgs_hr


    def load_batch_image(self, idx=0, img_paths=None, training=True, bicubic=False):
        """Loads a batch of images from datapath folder""" 

        # Starting index to look in
        cur_idx = 0
        
        if not img_paths:
            cur_idx = idx*self.batch_size 
        # Scale and pre-process images
        imgs_hr, imgs_lr = [], []
        while True:
            # Check if done with batch
            if img_paths is None:
                if cur_idx >= self.total_imgs:
                    cur_idx = 0
                if len(imgs_hr) >= self.batch_size:
                    break
            if img_paths is not None and len(imgs_hr) == len(img_paths):
                break            
            try: 
                # Load image
                img_hr = None
                if img_paths:
                    img_hr = self.load_img(img_paths[cur_idx],self.colorspace)
                else:
                    img_hr = self.load_img(self.img_paths[cur_idx],self.colorspace)
                # Create HR images to go through
                img_crops = []
                if training:
                    for i in range(self.crops_per_image):
                        #print(idx, cur_idx, "Loading crop: ", i)
                        img_crops.append(self.random_crop(img_hr, (self.height_hr, self.width_hr)))    
                else:
                    img_crops = [img_hr]
                # Downscale the HR images and save
                for img_hr in img_crops:

                    # TODO: Refactor this so it does not occur multiple times
                    if img_paths is None:
                        if cur_idx >= self.total_imgs:
                            cur_idx = 0
                        if len(imgs_hr) >= self.batch_size:
                            break
                    if img_paths is not None and len(imgs_hr) == len(img_paths):
                        break   

                    # For LR, do bicubic downsampling
                    lr_shape = (int(img_hr.shape[1]/self.scale), int(img_hr.shape[0]/self.scale))                      
                    img_lr = cv2.resize(img_hr,lr_shape, interpolation = cv2.INTER_CUBIC)

                    
                    # Scale color values
                    img_hr = self.scale_hr_imgs(img_hr)
                    img_lr = self.scale_lr_imgs(img_lr)

                    # Store images
                    #print(img_hr[:,:,:self.channels].shape,img_lr[:,:,:self.channels].shape)
                    imgs_hr.append(img_hr[:,:,:self.channels])
                    imgs_lr.append(img_lr[:,:,:self.channels])
                
            except Exception as e:
                print(e)
                pass
            finally:
                cur_idx += 1

        # Convert to numpy arrays when we are training 
        # Note: all are cropped to same size, which is not the case when not training
        if training:
            imgs_hr = np.array(imgs_hr)
            imgs_lr = np.array(imgs_lr)

        # Return image batch
        return imgs_lr, imgs_hr
    
    

def plot_test_images(model, loader, datapath_test, test_output, epoch, name='SRCNN', channels = 3,colorspace='RGB'):
    
    try:   
        # Get the location of test images
        test_images = [os.path.join(datapath_test, f) for f in os.listdir(datapath_test) if any(filetype in f.lower() for filetype in ['jpeg','mp4','264', 'png', 'jpg'])]
        
        # Load the images to perform test on images
        imgs_lr, imgs_hr = loader.load_batch(img_paths=test_images, training=False, bicubic=True)
        # Create super resolution and bicubic interpolation images
        imgs_sr = []
        imgs_bi = []
        srcnn_psnr = []
        bi_psnr = []
        for i in range(len(test_images)):
            
            pre=np.squeeze(
                    model.predict(
                        np.expand_dims(imgs_lr[i], 0),
                        batch_size=1
                    ),
                    axis=0
                )
            #pre[pre[:] > 255] = 255
            #pre[pre[:] < 0] = 0
            imgs_sr.append(pre)
 
        
        # Unscale colors values
        if channels == 1:
            imgs_lr = [loader.unscale_lr_imgs(img[:,:,0]).astype(np.uint8) for img in imgs_lr]
            imgs_hr = [loader.unscale_hr_imgs(img[:,:,0]).astype(np.uint8) for img in imgs_hr]
            imgs_sr = [loader.unscale_hr_imgs(img[:,:,0]).astype(np.uint8) for img in imgs_sr]
        else:
            if(colorspace == 'YCbCr'):
                imgs_lr = [cv2.cvtColor(loader.unscale_lr_imgs(img).astype(np.uint8), cv2.COLOR_YCrCb2BGR) for img in imgs_lr]
                imgs_hr = [cv2.cvtColor(loader.unscale_hr_imgs(img).astype(np.uint8), cv2.COLOR_YCrCb2BGR) for img in imgs_hr]
                imgs_sr = [cv2.cvtColor(loader.unscale_hr_imgs(img).astype(np.uint8), cv2.COLOR_YCrCb2BGR) for img in imgs_sr]
                
            else:
                imgs_lr = [loader.unscale_lr_imgs(img).astype(np.uint8) for img in imgs_lr]
                imgs_hr = [loader.unscale_hr_imgs(img).astype(np.uint8) for img in imgs_hr]
                imgs_sr = [loader.unscale_hr_imgs(img).astype(np.uint8) for img in imgs_sr]
	

        # Loop through images
        for img_hr, img_lr, img_sr, img_path in zip(imgs_hr, imgs_lr, imgs_sr, test_images):

            # Get the filename
            filename = os.path.basename(img_path).split(".")[0]
            
            # Bicubic upscale
            hr_shape = (int(img_hr.shape[1]), int(img_hr.shape[0]))                      
            img_bi = cv2.resize(img_lr,hr_shape, interpolation = cv2.INTER_CUBIC)
	    

            # Images and titles
            images = {
                'Low Resoluiton': [img_lr, img_hr],
                'Bicubic': [img_bi, img_hr],  
                name: [img_sr, img_hr], 
                'Original': [img_hr,img_hr]
            }
            srcnn_psnr.append(psnr(img_sr,img_hr,255.))
            bi_psnr.append(psnr(img_bi,img_hr,255.))
	    

        # Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1                    
            fig, axes = plt.subplots(1, 4, figsize=(40, 10))
            for i, (title, img) in enumerate(images.items()):
                axes[i].imshow(img[0])
                axes[i].set_title("{} - {} {}".format(title, img[0].shape, ("- psnr: "+str(round(psnr(img[0],img[1],255.),2)) if (title == name or title == 'Bicubic' ) else " ")))
                #axes[i].set_title("{} - {}".format(title, img.shape))
                axes[i].axis('off')
            plt.suptitle('{} - Epoch: {}'.format(filename, epoch))

            # Save directory                    
            savefile = os.path.join(test_output, "{}-Epoch{}.png".format(filename, epoch))
            fig.savefig(savefile)
            plt.close()
            gc.collect()
        print('test srcnn psnr: {} - test bi psnr: {}'.format(np.mean(srcnn_psnr),np.mean(bi_psnr)))
    except Exception as e:
        print(e)