from imutils import paths
from itertools import product
import numpy as np
import os, cv2

def getBaseFolders(base_path):
  return [os.path.join(base_path, folder) for folder in os.listdir(base_path) if os.path.isdir( os.path.join(base_path, folder))]

def resizeClassesImages(base_path, sizes):
  base_folders = getBaseFolders(base_path)
  sizes = list(sizes)
  for path, dimensions in list( product( base_folders, sizes ) ):
    print( "woking in", path, "with dimensions {}x{}".format(dimensions[0], dimensions[1]) )
    print("Get Images...")
    image_paths = list( paths.list_images( path ) )

    new_path = base_path + "{}x{}".format(dimensions[0], dimensions[1])
    new_path = os.path.join(new_path, os.path.basename(path))
    os.makedirs(new_path, exist_ok=True)
    new_image_paths = [os.path.join(new_path, os.path.basename(image_path)) for image_path in image_paths]

    print("Saving Images in '{}'... \n\n".format(new_path))
    for index in range(len(image_paths)):
      image = cv2.imread(image_paths[index])
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = cv2.resize(image, dimensions)
      cv2.imwrite(new_image_paths[index], image)
def imageGenerator(image_path, out_path):
  from numpy import expand_dims
  from keras.preprocessing.image import load_img
  from keras.preprocessing.image import img_to_array
  from keras.preprocessing.image import ImageDataGenerator
  
  image = cv2.imread(image_path)
  image = img_to_array(image)
  image = expand_dims(image, 0)
  datagen = ImageDataGenerator(brightness_range=[0.5,1.5], horizontal_flip=True, vertical_flip=True)
  it = datagen.flow(image, batch_size=1)
  batch = it.next()
  image = batch[0].astype('uint8')
  
  cv2.imwrite(out_path, image)
def maskImage(image_path, out_path):
  image = cv2.imread(image_path)
  image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  low_color = np.array([25, 52, 72])
  high_color = np.array([102, 255, 255])
  color_mask = cv2.inRange(image_hsv, low_color, high_color)
  masked_image = cv2.bitwise_and(image, image, mask=color_mask)

  cv2.imwrite(out_path, masked_image)
def maskClassesImages(base_path):
  base_folders = getBaseFolders(base_path)
  for path in base_folders:
    print( "woking in", path )
    print("Get Images...")
    image_paths = list( paths.list_images( path ) )
    
    print("Creating output list names...")
    new_path = base_path + "_masked"
    new_path = os.path.join(new_path, os.path.basename(path))
    os.makedirs(new_path, exist_ok=True)
    out_image_paths = [os.path.join(new_path, os.path.basename(image_path)) for image_path in image_paths]
    
    print("Conveting Images in '{}'... \n\n".format(new_path))
    for index in range(len(image_paths)):
      maskImage(image_paths[index], out_image_paths[index])

  print("Finish maskClassesImages!!!")
def classesImageGenerator(base_path, num_images=1):
  base_folders = getBaseFolders(base_path)
  for path in base_folders:
    print( "woking in", path )
    print("Get Images...")
    image_paths = list( paths.list_images( path ) )
    
    print("Creating output list names...")
    new_path = base_path + "_augmented"
    new_path = os.path.join(new_path, os.path.basename(path))
    os.makedirs(new_path, exist_ok=True)
    out_image_paths = [os.path.join(new_path, os.path.basename(image_path)) for image_path in image_paths]

    print("Generating Images in '{}'... \n\n".format(new_path))
    for index in range(len(image_paths)):
      for i in range(num_images):
        out_image_path = out_image_paths[index]
        dot_loc = out_image_path.rfind('.')
        out_image_path = out_image_path[:dot_loc] + " (0{}) ".format(i+1) + out_image_path[dot_loc:]
        imageGenerator(image_paths[index], out_image_path)

  print("Finish classesImageGenerator!!!")


if __name__ == '__main__':
  base_path = '..{}imagens'.format(os.sep)
  mask_base_path = base_path+"_masked"
  augmented_base_path = base_path+"_augmented"
  mask_augmented_base_path = mask_base_path+"_augmented"
  sizes = [(200, 150), (200, 267)]

  # resizeClassesImages(base_path=base_path, sizes=sizes)

  # print("Run maskClassesImages")
  # maskClassesImages(base_path=base_path)



  # print("Run classesImageGenerator")
  # classesImageGenerator(base_path=base_path, num_images=5)

  print("Run classesImageGenerator")
  classesImageGenerator(base_path=mask_base_path, num_images=5)



  # print("Run resizeClassesImages in", mask_base_path)
  # resizeClassesImages(base_path=mask_base_path, sizes=sizes)
  
  # print("Run resizeClassesImages in", augmented_base_path)
  # resizeClassesImages(base_path=augmented_base_path, sizes=sizes)
  
  # print("Run resizeClassesImages in", mask_augmented_base_path)
  # resizeClassesImages(base_path=mask_augmented_base_path, sizes=sizes)

  print("FINISH ALL!!!")