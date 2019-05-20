import cv2
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D, Lambda, MaxPooling2D
from keras.applications.vgg19 import VGG19
from keras.utils import data_utils as keras_utils
from keras.applications.vgg19 import preprocess_input
from skimage.measure import compare_psnr

class VGGLossNoActivation(object):
    """By ESRGAN a more effective perceptual loss constraining on features before activation rather than 
    after activation as practiced in SRGAN. 
    Reference: https://arxiv.org/abs/1809.00219"""

    def __init__(self, image_shape):
        self.model = self.create_model(image_shape)
        
    def create_model(self,image_shape):
        WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5')
        WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
        vgg19 = VGG19(include_top=False, weights=None, input_shape=image_shape)
        # Block 5 without activation relu
        x = Conv2D(512, (3, 3),
                      # activation='relu',
                      padding='same',
                      name='block5_conv4')(vgg19.get_layer('block5_conv3').output)
        #x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        model = Model(inputs=vgg19.input, outputs=x)
        weights_path = keras_utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path)
        model.trainable = False
        return model
    
    def preprocess_vgg(self, x):
        """Take a HR image [-1, 1], convert to [0, 255], then to input for VGG network"""
        if isinstance(x, np.ndarray):
            return preprocess_input((x+1)*127.5)
        else:            
            return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5))(x)
        
    # computes VGG loss or content loss
    def mse_content_loss(self, y_true, y_pred):
        return 1e-5 * K.mean(K.square(self.model(self.preprocess_vgg(y_true)) - self.model(self.preprocess_vgg(y_pred))),None)
    
    def euclidean_content_loss(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(self.model(self.preprocess_vgg(y_true)) - self.model(self.preprocess_vgg(y_pred))), axis=None))
    
    def plus_content_loss(self, y_true, y_pred):
        return (1e-5 * K.mean(K.square(self.model(self.preprocess_vgg(y_true)) - self.model(self.preprocess_vgg(y_pred))),None) + K.mean(K.square(y_pred - y_true), axis=None))



class VGGLoss(object):

    def __init__(self, image_shape):
        self.image_shape = image_shape
        self.vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        self.vgg19.trainable = False
        # Make trainable as False
        for l in self.vgg19.layers:
            l.trainable = False
        self.model = Model(inputs=self.vgg19.input, outputs=self.vgg19.get_layer('block5_conv4').output)
        self.model.trainable = False


    def preprocess_vgg(self, x):
        """Take a HR image [-1, 1], convert to [0, 255], then to input for VGG network"""
        if isinstance(x, np.ndarray):
            return preprocess_input((x+1)*127.5)
        else:            
            return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5))(x)
    
    # computes VGG loss or content loss
    def mse_content_loss(self, y_true, y_pred):
        return  K.mean(K.square(self.model(self.preprocess_vgg(y_true)) - self.model(self.preprocess_vgg(y_pred))))
    
    def euclidean_content_loss(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(self.model(self.preprocess_vgg(y_true)) - self.model(self.preprocess_vgg(y_pred))), axis=None))
    
    def plus_content_loss(self, y_true, y_pred):
        return (1. * K.mean(K.square(self.model(self.preprocess_vgg(y_true)) - self.model(self.preprocess_vgg(y_pred)))) + K.mean(K.square(y_pred - y_true), axis=None))


class Metrics():
    def __init__(self,use_gpu=True):
        '''
        net - ['squeeze','alex','vgg']
        '''

    def psnr_video(self,videopath1, videopath2):
        cap1 = cv2.VideoCapture(videopath1)
        cap2 = cv2.VideoCapture(videopath2)
        psnr_values = []
        while(cap1.isOpened() and cap2.isOpened()):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if(ret1 and ret2):
                psnr_values.append(self.calculate_psnr(frame1, frame2))        
            else:
                break
        cap1.release()
        cap2.release()
        return np.mean(psnr_values)
    
    def ssim_video(self,videopath1, videopath2):
        cap1 = cv2.VideoCapture(videopath1)
        cap2 = cv2.VideoCapture(videopath2)
        ssim_values = []
        while(cap1.isOpened() and cap2.isOpened()):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if(ret1 and ret2):
                ssim_values.append(self.calculate_ssim(frame1, frame2))        
            else:
                break
        cap1.release()
        cap2.release()
        return np.mean(ssim_values)

    # Calculate psnr between two images
    def calculate_psnr(self,img1, img2):
        # img1 and img2 have range [0, 255]
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))


    # Aux to Calculate psnr between two images
    def ssim(self,img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    # Calculate ssim between two images
    def calculate_ssim(self,img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return self.ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self.ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self.ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')


def unscale_hr_imgs(x):
    """Take a HR image [0, 1], convert to [0, 255]"""
    if isinstance(x, np.ndarray):
        return (x * 255)
    else:
        x = tf.keras.backend.cast(Lambda(lambda x: (x * 255))(x),'int32')            
        return tf.keras.backend.cast(x,'float32')

def ssim(y_true, y_pred,max_val = 255.):
    return tf.image.ssim(unscale_hr_imgs(y_true), unscale_hr_imgs(y_pred), max_val)

""" def psnr(y_true, y_pred, max_val = 255.):
    return tf.image.psnr(y_true, y_pred, 1.) """  

def psnr(y_true, y_pred, max_val = 1.):
    mse = mean_squared_error(y_true,y_pred,None)
    return 10.0 * (K.log(K.pow(max_val,2)/mse) /  K.log(10.0))

def psnr2(y_true, y_pred, max_val = 255.):
    return compare_psnr(y_true,y_pred,max_val)


def psnr3(y_true, y_pred, max_val = 255.):
    mse = mean_squared_error(unscale_hr_imgs(y_true),unscale_hr_imgs(y_pred),None)
    return 10.0 * (K.log(K.pow(max_val,2)/mse) /  K.log(10.0))

def mean_squared_error(y_true, y_pred,axis=None):
    return K.mean(K.square(y_pred - y_true), axis=axis)

def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.sum(y_true * y_pred, axis=-1)

def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)

def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)

def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return 100. * K.mean(diff, axis=-1)

def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)

def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)

def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)

def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

def euclidean_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=None))

# https://towardsdatascience.com/4-awesome-things-you-can-do-with-keras-and-the-code-you-need-to-make-it-happen-9b591286e4e0
def charbonnier(y_true, y_pred):
    epsilon = 1e-3
    error = y_true - y_pred
    p = K.sqrt(K.square(error) + K.square(epsilon))
    return K.mean(p)


# Aliases.

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_proximity
euclidean = euclidean_loss