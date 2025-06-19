import os
import random
import tensorflow 
import warnings

#validates packages required for environment
from package_validation import check_cuda,validate_requirements

# Components for building the layered CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, AveragePooling2D,MaxPooling2D, Dropout, Flatten, Dense,Activation

# The ImageDataGenerator will carry out the flow of image data from storage to memory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#check cuda and required packages are installed
check_cuda()
validate_requirements("./requirements.txt")


# Set the base directory for the dataset
# Assume the dataset is present and the folder structure matches what is expected
src = "./Dataset/PetImages/"

# Parition Dataset into training and testing set ---------------------------------------------------------------------------

# Import the function for splitting the dataset into the training set and the testing set
warnings.filterwarnings("ignore")
from dataset_utilities import train_test_split

# Create separate folders for storing the training samples and the test samples
#if not os.path.isdir(src + 'Train/'):

# Partition the dataset
print("Partitioning the dataset into the training set and testing set...")
train_test_split(src)


#Build & Compile Convolutional Neural Netowrk --------------------------------------------------------------------------------
# Define parameters
KERNEL_SIZE = 3                        # This is the sliding window that will scan an image and create a feature set
NUM_FILTERS_ONE = 64                  # 64 filters
NUM_FILTERS_TWO = 16                   # 16 filters for second layer
INPUT_SIZE  = 32                       # Compress image dimensions to 32 x 32 (may lose some data)
POOL_SIZE_ONE = 2                      # The stride is 2 so data is 75% smaller
POOL_SIZE_TWO = 4                      # The stride is 4 so data is 93.7% smaller final output is 4x4x16
BATCH_SIZE = 32                        # Use 32 training samples per batch 
STEPS_PER_EPOCH = 20000//BATCH_SIZE    # Number of iterations per epoch; the '//' floor nearest int
EPOCHS = 10                            # do 10 epochs

# Start with base sequential layer model
model = Sequential()


# Add first 2D convolutional layer; this layer reads the actual image files
# Input image is 3 channel (RGB) 32x32 length*width
model.add(Conv2D(filters=NUM_FILTERS_ONE, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), input_shape = (INPUT_SIZE, INPUT_SIZE, 3), strides=(1,1), padding='same'))

#Normalize the batch
model.add(BatchNormalization())

#Apply ReLU function to the resulting data
model.add(Activation('relu'))

# Add first subsampling layer using max pooling so reduces total area by 81% (7x7 output dim / 16x16 input dim)
model.add(MaxPooling2D(pool_size = (POOL_SIZE_ONE, POOL_SIZE_ONE),strides=(2,2)))

# Add second 2D convolutional layer; this layer reads the subsampled feature map from the first convolutional layer
model.add(Conv2D(filters= NUM_FILTERS_TWO, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), strides=(1,1), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))
# Add second subsampling layer, another reduction in 75%
model.add(MaxPooling2D(pool_size = (POOL_SIZE_TWO, POOL_SIZE_TWO),strides=(4,4)))

# Flatten the multidmensional vector received from the second subsampling layer into a one-dimensional vector
model.add(Flatten()) #128 neurons * 256 (shape of flattened vector)
model.add(Dense(units = 128, activation = 'relu'))

# Add dropout (turns off 30% of neurons)
model.add(Dropout(0.3))
# Add a dense fully connected layer with 64 nodes

# Add second fully connected layer, with only a single node and using a sigmoid activation function
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compile model using adam optimizer and binary cross-entropy loss function
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

#Training-------------------------------------------------------------------------------------

# Create ImageDataGenerator to load batches of images at a time into memory
training_data_generator = ImageDataGenerator(rescale = 1./255)

# Create batch-loaded training set
training_set = training_data_generator.flow_from_directory(src + 'Train/',
                                                target_size = (INPUT_SIZE, INPUT_SIZE),
                                                batch_size = BATCH_SIZE,
                                                class_mode = 'binary')

# Train the model
print("Training the CNN with AVG_POOLING=2, INPUT_SIZE=32 ...")
train_score=model.fit(training_set, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, verbose=1)
print(f"Training Set Metrics")
for metric, value in train_score.history.items():
    print(f"{metric}: {value[-1]}")

# Test the model ------------------------------------------------------------------------------

# Create ImageDataGenerator to load batches of images at a time into memory
testing_data_generator = ImageDataGenerator(rescale = 1./255)

# Create batch-loaded testing set
test_set = testing_data_generator.flow_from_directory(src + 'Test/', target_size = (INPUT_SIZE, INPUT_SIZE), batch_size = BATCH_SIZE, class_mode = 'binary')

# Test the model
print("Testing the CNN on new data...")
validation_score = model.evaluate(test_set, steps=100)

# Save the model and load the .h5 file for later use.


#Since i dont want to train the model on validation set data, i cant use .history on the metrics for .evaluate the model
#just map the matric names to what they should be, ie "compile_metrics" should be "accuracy"
metrics_map={
    "compile_metrics":"accuracy"
}
#Results --------------------------------------------------------------------------------------

# Display the results
print("RESULTS:")
test_accuracy=None
test_loss=None
for idx, metric in enumerate(model.metrics_names):
    #strange naming issue with "accuracy displaying as 'compile metrics, i just used a map and condition logic to fix"
    if metric in metrics_map:

        print("{}: {}".format(metrics_map[metric], validation_score[idx]))
        test_accuracy=validation_score[idx]
    else:
        print("{}: {}".format(metric, validation_score[idx]))
        test_loss=validation_score[idx]
try:
    model_name=f"cats_dogs_classifier_accuracy_{test_accuracy:.2f}_loss_{test_loss:.2f}.h5"
    model.save(model_name)
except Exception:
    raise 