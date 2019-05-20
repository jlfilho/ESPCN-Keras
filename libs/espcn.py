import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import ReLU
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.initializers import RandomNormal


from util import DataLoader, plot_test_images
from losses import psnr3 as psnr
from losses import euclidean

class SRCNN():
    def __init__(self,
                 height_lr=24, width_lr=24, channels=1,
                 upscaling_factor=4, lr = 1e-3,
                 training_mode=True,
                 media_type='i'
                 ):
        self.media_type = media_type

        # Low-resolution image dimensions
        self.height_lr = height_lr
        self.width_lr = width_lr

        # High-resolution image dimensions
        if upscaling_factor not in [2, 4, 8]:
            raise ValueError(
                'Upscaling factor must be either 2, 4, or 8. You chose {}'.format(upscaling_factor))
        self.upscaling_factor = upscaling_factor
        self.height_hr = int(self.height_lr * self.upscaling_factor)
        self.width_hr = int(self.width_lr * self.upscaling_factor)

        # Low-resolution and high-resolution shapes
        self.channels = channels
        self.shape_lr = (self.height_lr, self.width_lr, self.channels)
        self.shape_hr = (self.height_hr, self.width_hr, self.channels)

        self.loss = "mse"
        self.lr = lr

        self.srcnn = self.build_srcnn()
        self.compile_srcnn(self.srcnn)


    def save_weights(self, filepath):
        """Save the networks weights"""
        self.srcnn.save_weights(
            "{}_{}X.h5".format(filepath, self.upscaling_factor))
        

    def load_weights(self, weights=None, **kwargs):
        print(">> Loading weights...")
        if weights:
            self.srcnn.load_weights(weights, **kwargs)
        
    
    def compile_srcnn(self, model):
        """Compile the srcnn with appropriate optimizer"""
        
        model.compile(
            loss=self.loss,
            optimizer=Adam(lr=self.lr,beta_1=0.9, beta_2=0.999), #SGD(lr=self.lr, momentum=0.9, decay=1e-6, nesterov=True), 
            metrics=[psnr]
        )

    def build_srcnn(self):

        inputs = Input(shape=(None, None, self.channels))
          
        conv1 = Conv2D(filters= 64, kernel_size = (9,9), strides=1, 
            kernel_initializer=RandomNormal(mean=0.0, stddev=0.001, seed=None),bias_initializer='zeros',
            padding = "valid", use_bias=True, name='conv1')(inputs)
        conv1 = ReLU()(conv1)

        conv2 = Conv2D(filters= 32, kernel_size = (5,5), strides=1, 
            kernel_initializer=RandomNormal(mean=0.0, stddev=0.001, seed=None),bias_initializer='zeros',
            padding = "valid", use_bias=True, name='conv2')(conv1)
        conv2 = ReLU()(conv2)

        conv3 = Conv2D(filters= 1, kernel_size = (5,5), strides=1, 
            kernel_initializer=RandomNormal(mean=0.0, stddev=0.001, seed=None),bias_initializer='zeros',
            padding = "valid", use_bias=True, name='conv3')(conv1)
        
        model = Model(inputs=inputs, outputs=conv3)
        model.summary()
        return model

    def train(self,
            epochs=50,
            batch_size=8,
            steps_per_epoch=5,
            steps_per_validation=5,
            crops_per_image=4,
            print_frequency=5,
            log_tensorboard_update_freq=10,
            workers=4,
            max_queue_size=5,
            model_name='TSRGANSRCNN',
            datapath_train='../../../videos_harmonic/MYANMAR_2160p/train/',
            datapath_validation='../../../videos_harmonic/MYANMAR_2160p/validation/',
            datapath_test='../../../videos_harmonic/MYANMAR_2160p/test/',
            log_weight_path='../model/', 
            log_tensorboard_path='../logs/',
            log_test_path='../test/'
        ):

        # Create data loaders
        train_loader = DataLoader(
            datapath_train, batch_size,
            self.height_hr, self.width_hr,
            self.upscaling_factor,
            crops_per_image,
            self.media_type
        )

        validation_loader = None 
        if datapath_validation is not None:
            validation_loader = DataLoader(
                datapath_validation, batch_size,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                crops_per_image,
                self.media_type
        )

        test_loader = None
        if datapath_test is not None:
            test_loader = DataLoader(
                datapath_test, 1,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                1,
                self.media_type
        )

        # Callback: tensorboard
        callbacks = []
        if log_tensorboard_path:
            tensorboard = TensorBoard(
                log_dir=os.path.join(log_tensorboard_path, model_name),
                histogram_freq=0,
                batch_size=batch_size,
                write_graph=True,
                write_grads=True,
                update_freq=log_tensorboard_update_freq
            )
            callbacks.append(tensorboard)
        else:
            print(">> Not logging to tensorboard since no log_tensorboard_path is set")

        # Callback: Stop training when a monitored quantity has stopped improving
        earlystopping = EarlyStopping(
            monitor='val_loss', 
            patience=30, verbose=1, 
            restore_best_weights=True )
        callbacks.append(earlystopping)

        # Callback: Reduce lr when a monitored quantity has stopped improving
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=20, min_lr=2*1e-6)
        callbacks.append(reduce_lr)

        # Callback: save weights after each epoch
        modelcheckpoint = ModelCheckpoint(
            os.path.join(log_weight_path, model_name + '_{}X.h5'.format(self.upscaling_factor)), 
            monitor='val_loss', 
            save_best_only=True, 
            save_weights_only=True)
        callbacks.append(modelcheckpoint)
  
        # Callback: test images plotting
        if datapath_test is not None:
            testplotting = LambdaCallback(
                on_epoch_end=lambda epoch, logs: None if ((epoch+1) % print_frequency != 0 ) else plot_test_images(
                    self.srcnn,
                    test_loader,
                    datapath_test,
                    log_test_path,
                    epoch+1,
                    name=model_name))
        callbacks.append(testplotting)

        #callbacks.append(TQDMCallback())

        self.srcnn.fit_generator(
            train_loader,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_loader,
            validation_steps=steps_per_validation,
            callbacks=callbacks,
            shuffle=True,
            use_multiprocessing=workers>1,
            workers=workers
        )

# Run the SRCNN network
if __name__ == "__main__":

    # Instantiate the TSRGAN object
    print(">> Creating the SRCNN network")
    srcnn = SRCNN(height_lr=16, width_lr=16,lr=1e-5,upscaling_factor=2)
    #srcnn.load_weights(weights='../model/SRCNN_2X.h5')
    

    srcnn.train(
            epochs=1000,
            batch_size=32,
            steps_per_epoch=200,
            steps_per_validation=10,
            crops_per_image=4,
            print_frequency=1,
            log_tensorboard_update_freq=10,
            workers=2,
            max_queue_size=11,
            model_name='SRCNN',
            datapath_train='../../data/train2017/', 
            datapath_validation='../../data/val_large', 
            datapath_test='../../data/SR_testing_datasets/Set5/',
            log_weight_path='../model/', 
            log_tensorboard_path='../logs/',
            log_test_path='../test/'
    )

