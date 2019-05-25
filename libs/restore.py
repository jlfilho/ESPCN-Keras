import math
import imageio
import skvideo.io
import numpy as np
import cv2

from PIL import Image
from timeit import default_timer as timer
from util import DataLoader




def scale_lr_imgs(imgs):
    """Scale low-res images prior to passing to SRGAN"""
    return imgs / 255.
    
def unscale_hr_imgs(imgs):
    """Un-Scale high-res images"""
    #return (imgs + 1.) * 127.5
    return imgs * 255.

    

def sr_genarator(model,img_lr,scale):
    """Predict sr frame given a LR frame"""
    # Predict high-resolution version (add batch dimension to image)
    img_lr = scale_lr_imgs(img_lr)
    img_sr = model.predict(np.expand_dims(img_lr, 0))
    img_sr = unscale_hr_imgs(img_sr)
    img_sr[img_sr[:] > 255] = 255
    img_sr[img_sr[:] < 0] = 0
    # Remove batch dimension
    img_sr = img_sr.reshape(img_sr.shape[1], img_sr.shape[2], img_sr.shape[3])
    return img_sr


def write_srvideo(model=None,lr_videopath=None,sr_videopath=None,scale=None,print_frequency=False,crf=15,fps=None):
    """Predict SR video given LR video """
    # start the FFmpeg writing subprocess with following parameters
    videogen = skvideo.io.FFmpegReader(lr_videopath)
    t_frames = videogen.getShape()[0] 
    metadata = skvideo.io.ffprobe(lr_videopath)
    _fps = metadata['video']['@r_frame_rate'] if (fps == None) else str(fps) 
    writer = skvideo.io.FFmpegWriter(sr_videopath, outputdict={
        '-vcodec': 'libx264','-crf': str(crf), '-r': _fps })
    count = 0
    time_elapsed = []
    print(">> Writing video...")
    for frame in videogen:
        start = timer()
        img_sr = sr_genarator(model,frame,scale=scale)
        writer.writeFrame(img_sr)
        end = timer()
        time_elapsed.append(end - start)
        count +=1
        if (print_frequency): 
            if(count % print_frequency == 0):
                print('... Time per Frame: '+str(np.mean(time_elapsed))+'s')
                print('... Estimated time: '+str(np.mean(time_elapsed)*(t_frames-count)/60.)+'min')
    writer.close()
    print('>> Video resized in '+str(np.sum(time_elapsed))+'s')
    return time_elapsed


def write_sr_images(model=None, lr_imagepath=None, sr_imagepath=None,scale=None):
    print(">> Writing image...")
    time_elapsed = []
    # Load the images to perform test on images
    img_lr = DataLoader.load_img(lr_imagepath,colorspace='RGB')
        
    # Create super resolution images
    start = timer()
    img_sr = sr_genarator(model,img_lr,scale)    
    end = timer()
    time_elapsed.append(end - start)   

    img_sr = Image.fromarray(img_sr.astype(np.uint8))
    img_sr.save(sr_imagepath)
    print('>> Image resized in '+str(np.mean(time_elapsed))+'s')
    return time_elapsed