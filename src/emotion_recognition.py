import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Set up command line argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")  # Add argument for mode selection
mode = ap.parse_args().mode                     # Get the mode from command line

def plot_model_history(model_history):
    """
    Function to plot training history
    Creates two subplots: one for accuracy and one for loss
    Saves the plot to 'plot.png' and displays it
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))  # Create 1x2 subplot

    # Plot accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),
                model_history.history['accuracy'])         # Training accuracy
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),
                model_history.history['val_accuracy'])     # Validation accuracy
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),
                      len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')

    # Plot loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),
                model_history.history['loss'])            # Training loss
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),
                model_history.history['val_loss'])        # Validation loss
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),
                      len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')

    fig.savefig('plot.png')
    plt.show()

# Define data directories and parameters
train_dir = 'data/train'           # Directory containing training images
val_dir = 'data/test'             # Directory containing validation images
num_train = 28709                 # Number of training samples
num_val = 7178                    # Number of validation samples
batch_size = 64                   # Batch size for training
num_epoch = 50                    # Number of training epochs

# Set up data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values to [0,1]
val_datagen = ImageDataGenerator(rescale=1./255)    # Same normalization for validation

# Configure training data generator
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),        # Resize images to 48x48
        batch_size=batch_size,
        color_mode="grayscale",     # Convert images to grayscale
        class_mode='categorical')    # Use categorical labels

# Configure validation data generator
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Create the CNN model
model = Sequential()

# First convolutional block
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Second convolutional block
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fully connected layers
model.add(Flatten())                # Flatten the 2D features
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 emotion classes

# Training mode
if mode == "train":
    # Compile model with categorical crossentropy loss and Adam optimizer
    model.compile(loss='categorical_crossentropy',
                 optimizer=Adam(lr=0.0001, decay=1e-6),
                 metrics=['accuracy'])

    # Train the model
    model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)

    plot_model_history(model_info)  # Plot training history
    model.save_weights('model.h5')  # Save model weights

# Display mode - real-time emotion detection
elif mode == "display":
    model.load_weights('model.h5')  # Load trained weights

    cv2.ocl.setUseOpenCL(False)    # Disable OpenCL to prevent issues

    # Dictionary mapping emotion indices to labels
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
                   4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Start webcam capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()     # Read frame from webcam
        if not ret:
            break

        # Load face detection cascade classifier
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

            # Extract and preprocess face region
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray, (48, 48)), -1), 0)

            # Predict emotion
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

            # Display emotion label
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Video', cv2.resize(frame,(1600,960),
                                     interpolation = cv2.INTER_CUBIC))

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
