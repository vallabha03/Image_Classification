from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import tensorflow as tf



# Define the number of classes in your dataset
num_classes = 9  # Replace with the actual number of classes

# Define data augmentation transformations
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and preprocess training data
train_data = train_datagen.flow_from_directory(
    "D:\\Image_project\\Image_classification\\image_classification\\archive\\images",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Build the model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_data,
    epochs=10,
    steps_per_epoch=len(train_data),
    verbose=1
)

# Evaluate on test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_data = test_datagen.flow_from_directory(
    "D:\\Image_project\\Image_classification\\image_classification\\archive\\images",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
loss, accuracy = model.evaluate(test_data, steps=len(test_data), verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Prediction function
def predict_fruit(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    
    class_label = None
    for label, idx in train_data.class_indices.items():
        if idx == class_idx:
            class_label = label
            break
    
    return class_label

# Test the prediction function
image_path = "D:\\Image_project\\Image_classification\\image_classification\\berry.jpg"
predicted_class = predict_fruit(image_path)
print(f"Predicted Class: {predicted_class}")
