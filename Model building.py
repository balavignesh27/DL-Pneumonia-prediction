from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from glob import glob
import matplotlib.pyplot as plt


# Resize the input image
image_size = [224,224]
train_path = 'Dataset/Train'
valid_path = 'Dataset/Test'

# Add preprocessing layer infront of VGG19
vgg = VGG19(input_shape=image_size+[3], weights='imagenet', include_top=False)

# Don't train the existing weights
for layer in vgg.layers:
    layer.trainable=False

# To get number of folder under Train folder
folders = glob('Dataset/Train/*')

# Add Flatten layer
x = Flatten()(vgg.output)

# Add Dense layer
prediction = Dense(len(folders), activation='softmax')(x)

# Create the model object
model = Model(inputs=vgg.input, outputs=prediction)

# View the structure of the model
model.summary()

# Compile the model by defining the optimizer and loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Preprocessing the input images for Train and Test
training_datagen = image.ImageDataGenerator(rescale=1./255,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True)
train_dataset = training_datagen.flow_from_directory('Dataset/Train',
                                                     target_size=(224,224),
                                                     batch_size=20,
                                                     class_mode='categorical')

testing_datagen = image.ImageDataGenerator(rescale=1./255)
test_dataset = testing_datagen.flow_from_directory('Dataset/Test',
                                                target_size=(224,224),
                                                batch_size=20,
                                                class_mode='categorical')

# Fit the model
r = model.fit(train_dataset, validation_data=test_dataset, epochs=5,
                        steps_per_epoch=len(train_dataset), validation_steps=len(test_dataset))

# Plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# Plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccuracyVal_acc')

# Save the model
model.save('model_vgg19.h5')


