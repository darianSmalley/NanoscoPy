import os
import matplotlib.pyplot as plt
import numpy as np
from shutil import copy
import random
import math

import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf 
from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

if tf.test.gpu_device_name(): 
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
#     print(device_lib.list_local_devices())
else:
    print("Please install GPU version of TF")
    
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# UTILITIY FUNCTIONS
def copy_all(sample, src_dir, dst_dir, label):
    try:
        for index, file in enumerate(sample):
            src_path = os.path.join(src_dir,file)
            dst_path = os.path.join(dst_dir,'{}.{}.jpg'.format(label,index))
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            copy(src_path, dst_path)

    except Exception as e:
        raise e
        print("Error! No files found here")

    print('Complete.')
    
# This function will plot images in the form of a grid with 1 row and 5 columns where 
# images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
    
def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues)
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    print('True Negatives: ', cm[0][0])
    print('False Positives: ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])

def plot_roc(name, labels, predictions, **kwargs):
    lw = 2
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    plt.plot(fp, tp, label=name, linewidth=lw, **kwargs)
    plt.xlabel('False positives (%)')
    plt.ylabel('True positives (%)')
    plt.gca().set_aspect('equal')
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for binary data')
    plt.show()

if __name__ == "__main__":
    labels = ['atomic-lattice-clear',
             'herringbone-clear']
    root_dir ='H:\\Research\\AI_STM\\sxm-storage\\'
    img_dir =  os.path.join(root_dir, 'All Au(111)')
    sources = [os.path.join(img_dir, label) for label in labels]
    num_data = np.array([len(os.listdir(source)) for source in sources])
    neg=num_data[0]
    pos=num_data[1]
    total_data = np.sum(num_data)
    precentage = 100.0*num_data/total_data
    plt.bar(range(len(num_data)), num_data)
    plt.show()
    print(precentage)
    num_train = np.array([math.floor(count * 0.75) for count in num_data])
    num_val =  np.array([math.floor(count * 0.25) for count in num_data])
    total_train = sum(num_train)
    total_val = sum(num_val)
    print("Train:", total_train, num_train)
    print("Val:", total_val, num_val)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, num_train, width, label='Train')
    rects2 = ax.bar(x + width/2, num_val, width, label='Test')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('Dataset Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.xticks(rotation=60)
    plt.show()
    
    out = '-'.join(labels)
    out_dir = os.path.join(root_dir, 'datasets', out)
    train_dir = os.path.join(out_dir, 'train')
    validation_dir = os.path.join(out_dir, 'validation')
    training_label_dirs = list(map(lambda label: os.path.join(train_dir, label), labels))
    validation_label_dirs = list(map(lambda label: os.path.join(validation_dir, label), labels))
    print(validation_label_dirs)
    
    # Re-order train and vlidation images
    FLAG = False
    if FLAG:
        for i, label in enumerate(labels):
            src = sources[i]
            train = training_label_dirs[i]
            n_train = num_train[i]
            val = validation_label_dirs[i]
            n_val = num_val[i]
            images = os.listdir(src)
            # Obtain a random list of images of size k without replacement to use as the valiation set
            val_sample = random.sample(images, k=n_val)
            # Remove sample from origional list 
            train_sample = list(set(images)-set(val_sample))
            # Copy validation images to new folder
            copy_all(val_sample, src, val, label)
            # Copy remaining training images to new folder
            copy_all(train_sample, src, train, label)
            
    BATCH_SIZE = 32
    IMG_SHAPE  = 256    
    image_generator = ImageDataGenerator(rescale=1./255,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.4,
                                        horizontal_flip=True,
                                        fill_mode='wrap',
                                        validation_split=0.2)

    train_data_gen = image_generator.flow_from_directory(subset='training',
                                                        directory=train_dir,
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        target_size=(IMG_SHAPE,IMG_SHAPE),
                                                        class_mode='binary')

    val_data_gen = image_generator.flow_from_directory(subset='validation',
                                                     directory=train_dir,
                                                     batch_size=BATCH_SIZE,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE),
                                                     class_mode='binary')

    image_gen_test = ImageDataGenerator(rescale=1./255)
    test_data_gen = image_gen_test.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=validation_dir,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE),
                                                     class_mode='binary')
    
    METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
    ]
    
    num_classes = len(labels)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, 7, activation='relu', input_shape=(256, 256, 3),  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Conv2D(8, 7, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(momentum=0.0),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(16,5, activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Conv2D(8, 1, activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Conv2D(16, 5, activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(momentum=0.0),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(32, (3,3), activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(momentum=0.0),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(16, 1, activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(momentum=0.0),

        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalAccuracy(),
              metrics=METRICS)
    model.summary()
    
    # TRAIN MODEL
    class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=np.unique(test_data_gen.classes), 
                                                  y=test_data_gen.classes)
    class_weights = dict(zip(range(len(class_weights)), class_weights))
    print(class_weights)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', 
                                                    verbose=1,
                                                    patience=10,
                                                    mode='max',
                                                    restore_best_weights=True)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # Test fit
    epochs=20
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch = train_data_gen.samples // BATCH_SIZE,
        epochs=epochs,
        validation_data =val_data_gen,
        validation_steps = val_data_gen.samples // BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[tensorboard_callback]
    )
    
    plot_metrics(history)
    
    # TEST MODEl
    train_predictions_baseline = model.predict_generator(train_data_gen, steps=BATCH_SIZE)
    test_predictions_baseline = model.predict_generator(test_data_gen, steps=BATCH_SIZE)
    plot_cm(test_data_gen.classes, test_predictions_baseline)
    plt.figure(figsize=(8, 6))
    plot_roc("Train Baseline", train_data_gen.classes, train_predictions_baseline)
    plot_roc("Test Baseline", test_data_gen.classes, test_predictions_baseline, linestyle='--')
    plt.legend(loc='lower right')
    plt.show()