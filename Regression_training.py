import os
import sys
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import callbacks
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras import backend as K
import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
import scipy

def main():
    start_time = time.time()
    input_base = '/srv/home/chenyu/CNN/Data/STO_100nm_PACBED/'
    input_sub_folder = ['0_0/','0.5_0.5/','0.25_0.25/','1_0/','1_1/','2_0/','2_2/','3_0/']    
    result_path =  '/srv/home/chenyu/CNN/Regression/Coarse_CNN/PartiallyTraining/35nm_512_dense_layer/attempt_1/'

    x_train_list = []
    y_train_list = []

    # training within certain integration range
    min_radius = 6
    max_radius = 9

    sx, sy = 0, 0

    for current_folder in input_sub_folder:
        input_folder = input_base + current_folder
        input_images = [image for image in os.listdir(input_folder) if 'STO' in image]

        for image in input_images:
            cmp = image.split('_')
            label = (float(cmp[1]))/17
            radius = int(cmp[4][0:-4])

            if radius >= min_radius and radius <= max_radius and label <= 1:

                img = np.load(input_folder + image).astype(dtype=np.float64)
                img = scale_range(img,0,1)
                img = img.astype(dtype=np.float32)
                img_size = img.shape[0]
                sx, sy = img.shape[0], img.shape[1]
                new_channel = np.zeros((img_size, img_size))
                img_stack = np.dstack((img, new_channel, new_channel))

                x_train_list.append(img_stack)
                y_train_list.append(label)

    nb_train_samples = len(x_train_list)
    print('Image loaded')
    print('input shape: ')
    print(sx, sy)
    print('training number: ')
    print(nb_train_samples)
    nb_class = len(set(y_train_list))
    x_train = np.concatenate([arr[np.newaxis] for arr in x_train_list])
    #y_train = to_categorical(y_train_list, num_classes=nb_class)
    y_train = np.asarray(y_train_list,dtype=np.float32)
    print('Size of image array in bytes')
    print(x_train.nbytes)
    np.save(result_path + 'y_train.npy', y_train)


    logs = [log for log in os.listdir(result_path) if 'log' in log]
    max_index = 0
    for log in logs:
        cur = int(log.split('_')[1])
        if cur > max_index:
            max_index = cur
    max_index = max_index + 1

    batch_size = 32
    # step 1
    save_bottleneck_features(x_train, y_train, batch_size, nb_train_samples,result_path)

    # step 2
    epochs = 12
    batch_size = 32  # batch size 32 works for the fullsize simulation library which has 19968 total files, total number of training file must be integer times of batch_size
    train_top_model(y_train, nb_class, max_index, epochs, batch_size, input_folder, result_path)

    # step 3
    epochs = 50
    batch_size = 32
    fine_tune(x_train, y_train, sx, sy, max_index, epochs, batch_size, input_folder, result_path)

    # step 4: test trained CNN on experiment data
    prediction(result_path)

    print('Total computing time is: ')
    print(int((time.time() - start_time) * 100) / 100.0)


def save_bottleneck_features(x_train, y_train, batch_size, nb_train_samples,result_path):
    model = applications.VGG16(include_top=False, weights='imagenet')
    print('before featurewise center')
    
    datagen = ImageDataGenerator(
        featurewise_center=True,
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=1,
        vertical_flip=1,
        shear_range=0.05)
 

    datagen = ImageDataGenerator(
        featurewise_center=True)

    datagen.fit(x_train)
    print('made it past featurewise center')
    generator = datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=False)
    print('made it past generator')

    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    print('made it past the bottleneck features')
    np.save(result_path + 'bottleneck_features_train.npy',
            bottleneck_features_train)

def train_top_model(y_train, nb_class, max_index, epochs, batch_size, input_folder, result_path):
    train_data = np.load(result_path + 'bottleneck_features_train.npy')
    train_labels = y_train
    print(train_data.shape, train_labels.shape)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation=None))

    # compile setting:
    lr = 0.05
    decay = 5e-5
    momentum = 0.9
    optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    loss = 'mse'
    model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])
    
    bottleneck_log = result_path + 'training_' + str(max_index) + '_bnfeature_log.csv'
    csv_logger_bnfeature = callbacks.CSVLogger(bottleneck_log)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=30, verbose=1, mode='auto')

    model.fit(train_data,train_labels,epochs=epochs,batch_size=batch_size,shuffle=True,
            callbacks=[csv_logger_bnfeature, earlystop],verbose=2,validation_split=0.2)

    with open(bottleneck_log, 'a') as log:
        log.write('\n')
        log.write('input images: ' + input_folder + '\n')
        log.write('batch_size:' + str(batch_size) + '\n')
        log.write('learning rate: ' + str(lr) + '\n')
        log.write('learning rate decay: ' + str(decay) + '\n')
        log.write('momentum: ' + str(momentum) + '\n')
        log.write('loss: ' + loss + '\n')

    model.save_weights(result_path + 'bottleneck_fc_model.h5')

def fine_tune(train_data, train_labels, sx, sy, max_index, epochs, batch_size, input_folder, result_path):
    print(train_data.shape, train_labels.shape)

    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(sx, sy, 3))
    print('Model loaded')

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.3))
    top_model.add(Dense(1,activation=None))

    top_model.load_weights(result_path + 'bottleneck_fc_model.h5')

    new_model = Sequential()
    for l in model.layers:
        new_model.add(l)
    new_model.add(top_model)

    # compile settings
    lr = 0.001
    decay = 1e-5
    momentum = 0.9
    optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    loss = 'mse'
    new_model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])

    fineture_log = result_path + 'training_' + str(max_index) + '_finetune_log.csv'
    csv_logger_finetune = callbacks.CSVLogger(fineture_log)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')

    datagen = ImageDataGenerator(
        featurewise_center=True,
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=1,
        vertical_flip=1,
        shear_range=0.05)

    datagen.fit(train_data)

    generator = datagen.flow(
        train_data,
        train_labels,
        batch_size=batch_size,
        shuffle=True)

    validation_generator = datagen.flow(
        train_data,
        train_labels,
        batch_size=batch_size,
        shuffle=True)

    new_model.fit_generator(generator,epochs=epochs,steps_per_epoch=len(train_data) / 32,validation_data=validation_generator,validation_steps=(len(train_data)//5)//32,
            callbacks=[csv_logger_finetune, earlystop],verbose=2)

    #new_model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2,
                  #callbacks=[csv_logger_finetune, earlystop])

    with open(fineture_log, 'a') as log:
        log.write('\n')
        log.write('input images: ' + input_folder + '\n')
        log.write('batch_size:' + str(batch_size) + '\n')
        log.write('learning rate: ' + str(lr) + '\n')
        log.write('learning rate decay: ' + str(decay) + '\n')
        log.write('momentum: ' + str(momentum) + '\n')
        log.write('loss: ' + loss + '\n')

    new_model.save(result_path + 'FinalModel.h5')  # save the final model for future loading and prediction

# Use single image series to test the trained CNN and check whether it works on experiment dataset
def prediction(model_path):

    case = 'fine classification'
    input_base = '/srv/home/chenyu/DEbackup/033119/S*/'
    radius_list = [1,3,5,7,9,11]

    # Load model:
    model = load_model(model_path + 'FinalModel.h5')

    # Set up figure:
    fig = plt.figure(figsize=(10,10))
    fig.add_subplot(111)
    plt.title('CNN Prediction vs Truth',fontsize=26)

    # Loop over all radii
    for iradius in range(len(radius_list)):
        radius = radius_list[iradius]
        x_train_list = []
        y_train_list = []

        # Loop over all folders (thicknesses)
        for ifolder in range(len(glob(input_base))):
            input_folder = glob(input_base)[ifolder]
            input_images = [image for image in os.listdir(input_folder) if 'radius_' + str(radius) + '.npy' in image]
            # Confirm this folder has and only has one file related to target radius:
            if len(input_images)==1:
                input_images = input_images[0]

                img_array = np.load(input_folder + input_images)
                label_array = np.load(input_folder + 'SrPeaks_thickness.npy')   # in unit of unit cells

                # prepare data to be looped through
                img_array = img_array.astype('double')
                im_range = img_array.shape[2]
                for i in range(im_range):
                    # prepare individual array to be formatted for prediction
                    img = np.squeeze(img_array[:,:,i])
                    img = scipy.misc.imresize(img, (157, 157))
                    img = img.astype('double')
                    img = scale_range(img, 0, 1)

                    # add image and labels to prediction arrays
                    img_size = img.shape[0]
                    new_channel = np.zeros((img_size, img_size))
                    img_stack = np.dstack((img, new_channel, new_channel))
                    label = label_array[i]

                    x_train_list.append(img_stack)
                    y_train_list.append(label)

    # Run prediction for each integration radius
        datagen = ImageDataGenerator(featurewise_center=True)
        datagen.fit(np.asarray(x_train_list))
        generator = datagen.flow(np.asarray(x_train_list), batch_size=32)
        p_classes_2 = model.predict_generator(generator)

    # Plot results for each integration radius

        if 'classification' in case:
            if 'fine' in case:
                # Print clasification results, fine CNN
                p_classes = model.predict_classes(np.asarray(x_train_list), batch_size=32)  # for classification
                p_class_list = np.asarray(p_classes)
                y_list = np.asarray(y_train_list)
                print(p_class_list * 0.3905)    # results in nm
                print(np.average(p_class_list * 0.3905))
                print(y_list * 0.3905)          # truth in nm
                plt.scatter(y_list * 0.3905, p_class_list * 0.3905, label = 'Radius = ' + str(radius))
            elif 'coarse' in case:
                p_classes = model.predict_classes(np.asarray(x_train_list), batch_size=32)  # for classification
                p_class_list = np.asarray(p_classes)
                y_list = np.asarray(y_train_list)
                print(p_class_list * 5 * 0.3905)    # results in nm
                print(np.average(p_class_list * 5 * 0.3905))
                print(y_list * 0.3905)          # truth in nm
                plt.scatter(y_list * 0.3905, p_class_list * 0.3905 * 5, label = 'Radius = ' + str(radius))

        elif 'regression' in case:
            if 'fine' in case:
                # Print regression results, fine CNN
                p_arrays = model.predict(np.asarray(x_train_list), batch_size=32)   # For regression
                print(p_arrays * 90 * 0.3905)    # result in nm
                print(np.average(p_arrays * 90 * 0.3905))
                print(y_list * 0.3905)  # truth in nm
            elif 'coarse' in case:
                # Print regression results, coarse CNN
                p_arrays = model.predict(np.asarray(x_train_list), batch_size=32)   # For regression
                print(p_arrays * 51 * 2)    # result in nm
                print(np.average(p_arrays * 51 * 2))
                print(np.average(np.asarray(y_list * 0.3905)))  # truth in nm

        
            
    # Add legends and labels after all radii processed
    plt.legend()
    plt.xlabel('Truth',fontsize=20)
    plt.ylabel('CNN Prediction', fontsize=20)
    plt.xlim([0,100])
    plt.ylim([0,100])
    fig.savefig('Validation.png')


def scale_range(input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input
 

# step 4 make predictions using experiment results

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1])
    main()
