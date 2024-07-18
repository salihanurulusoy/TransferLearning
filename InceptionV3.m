import os
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define the directories
train_dir = '.../TrainSet'
test_dir = '.../TestSet'
validation_dir = '.../ValidationSet'
numClasses = 2  # If you have two classes

# Define grid of hyperparameters
param_grid = {
    'learning_rate': [0.1,0.01,0.001],
    'batch_size': [16,32,64],
}

best_accuracy = 0
best_params = None

# Perform grid search
for params in ParameterGrid(param_grid):
    # Load pre-trained InceptionV3 model for each grid search iteration
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    # Create data generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=params['batch_size'],
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=params['batch_size'],
        class_mode='categorical'
    )

    # Define the model architecture to accept raw images as input
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(numClasses, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    # Compile the model with current hyperparameters
    model.compile(optimizer=SGD(learning_rate=params['learning_rate']), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_inceptionv3.h5', monitor='val_accuracy', save_best_only=True, mode='max')

    # Train the model with early stopping and model checkpoint
    num_epochs = 100  # You can adjust the number of epochs here
    history = model.fit(
        train_generator,
        epochs=num_epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, model_checkpoint],  # Include early stopping and model checkpoint callbacks
        verbose=1
    )

    # Evaluate the model on validation data
    _, val_accuracy = model.evaluate(validation_generator, verbose=0)

    # Update best accuracy and best parameters if necessary
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = params

# Print best parameters and accuracy
print('Best parameters:', best_params)
print('Best validation accuracy:', best_accuracy)

# Load the best model for evaluation
best_model = load_model('best_inceptionv3.h5')

# Create data generator for test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=params['batch_size'],
    class_mode='categorical',
    shuffle=False  # Important to keep filenames in the same order as predictions
)

# Predict classes for test data
Y_pred_test_prob = best_model.predict(test_generator)
Y_pred_test = np.argmax(Y_pred_test_prob, axis=1)

# Calculate evaluation metrics
test_accuracy = accuracy_score(test_generator.classes, Y_pred_test)
test_precision = precision_score(test_generator.classes, Y_pred_test, average='weighted')
test_recall = recall_score(test_generator.classes, Y_pred_test, average='weighted')
test_f1 = f1_score(test_generator.classes, Y_pred_test, average='weighted')

print('Test accuracy:', test_accuracy)
print('Test precision:', test_precision)
print('Test recall:', test_recall)
print('Test F1-score:', test_f1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_generator.classes, Y_pred_test)

print('Confusion Matrix:')
print(conf_matrix)
