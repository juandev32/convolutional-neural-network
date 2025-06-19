#imports
import os				# for functions that depend on the operating system
import shutil			# allows high-level file operations such as removal of files an folder hierarchical trees
import random			# for random numbers
import piexif			# for dealing with image metadata ; remove() to remove corrupted exif metadata from image files

#splits the dataset, creating the training set and testing set; removes files that are corrupt, not an image file, or should not be used.                                        #####

def train_test_split(src_folder, train_size = 0.8):
	"""
	Random test/train images requires removing all previous images and redefining path during every run
	without the next 4 lines, the images from previous run will be left in folder
	the following 4 lines required for redefining path
	"""
	# remove existing training and testing subfolders
	# rmtree() removes directory, so every run will create a new directory  in
	shutil.rmtree(src_folder + 'Train/Cat/', ignore_errors=True)
	shutil.rmtree(src_folder + 'Train/Dog/', ignore_errors=True)
	shutil.rmtree(src_folder + 'Test/Cat/', ignore_errors=True)
	shutil.rmtree(src_folder + 'Test/Dog/', ignore_errors=True)

	# create new empty train and test folders
	# recursivly creates new directiory, from sourceFolderDirectory/{path}
	os.makedirs(src_folder + 'Train/Cat/')
	os.makedirs(src_folder + 'Train/Dog/')
	os.makedirs(src_folder + 'Test/Cat/')
	os.makedirs(src_folder + 'Test/Dog/')

	# retrieve the cat images
	# os.walk returns root {name of base folder}, dir {list of subfoler names}, files {list of file names}
	# only need list of cat images 
	_, _, cat_images = next(os.walk(src_folder + 'Cat/'))

	"""
	removing specific files
		best to preserve raw data and any alterations should be done in the program 
		helps with replicabiliy? So that you dont need to manualy delete files if redownloading data set
		best practice to preserve raw dataset just incase
	"""
	# these files are corrupt or they are non-image files	
	files_to_be_removed = ['Thumbs.db', '666.jpg', '835.jpg']
	
	# remove unwanted files
	for file in files_to_be_removed:
		cat_images.remove(file)
	
	# calculate number of cat images left, and determine the number
	# of cat images to use for the training and testing sets
	# sizing of training set is .8 * numImages
	num_cat_images = len(cat_images)
	num_cat_images_train = int(train_size * num_cat_images)
	num_cat_images_test = num_cat_images - num_cat_images_train

	# now retrieve the dog images
	_, _, dog_images = next(os.walk(src_folder+'Dog/'))

	# these files are corrupt or they are non-image files
	files_to_be_removed = ['Thumbs.db', '11702.jpg']
	
	# remove unwanted files
	for file in files_to_be_removed:
		dog_images.remove(file)

	# calculate number of dog images left, and determine the number
	# of dog images to use for the training and testing sets
	num_dog_images = len(dog_images)
	num_dog_images_train = int(train_size * num_dog_images)
	num_dog_images_test = num_dog_images - num_dog_images_train

	"""
	Sample dataset for training sets and testing sets
	"""

	# randomly assign cat images to the training set
	cat_train_images = random.sample(cat_images, num_cat_images_train)

	for img in cat_train_images:
		shutil.copy(src=src_folder + 'Cat/' + img, dst=src_folder + 'Train/Cat/')

	# place leftover cat images in the testing set
	cat_test_images  = [img for img in cat_images if img not in cat_train_images]
	
	for img in cat_test_images:
		shutil.copy(src=src_folder + 'Cat/' + img, dst=src_folder + 'Test/Cat/')

	# randomly assign dog images to the training set
	dog_train_images = random.sample(dog_images, num_dog_images_train)

	for img in dog_train_images:
		shutil.copy(src=src_folder + 'Dog/' + img, dst=src_folder + 'Train/Dog/')
	
	# place leftover dog images in the testing set
	dog_test_images  = [img for img in dog_images if img not in dog_train_images]
	
	for img in dog_test_images:
		shutil.copy(src=src_folder + 'Dog/' + img, dst=src_folder + 'Test/Dog/')




	# remove corrupted exif data from the dataset
	remove_exif_data(src_folder + 'Train/')
	remove_exif_data(src_folder + 'Test/')


# remove corrupt exif data from Microsoft's dataset
def remove_exif_data(src_folder):
	_, _, cat_images = next(os.walk(src_folder + 'Cat/'))

	for img in cat_images:
		try:
			piexif.remove(src_folder + 'Cat/' + img)
		except:
			pass

	_, _, dog_images = next(os.walk(src_folder + 'Dog/'))

	for img in dog_images:
		try:
			piexif.remove(src_folder + 'Dog/' + img)
		except:
			pass

