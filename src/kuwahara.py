# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from skimage import color
import argparse
import cv2
from tqdm import tqdm

def parse_args():
  cmd = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  cmd.add_argument('-o', dest='outfile', nargs='?', help='output file')
  cmd.add_argument('-i', dest='index', type=int, default=1, help='index for finding images to read')
  return cmd.parse_args()

# Read source, target and mask for a given id
def read_img(id, path = ""):
  img = plt.imread(path + "img_" + id + ".jpg")
  info = np.iinfo(img.dtype) # get information about the image type (min max values)
  img = img.astype(np.float32) / info.max # normalize the image into range 0 and 1
  return img

def kuwahara_basic(img, kernel_size):
  output = np.empty((img.shape))
  radius = kernel_size // 2
  quadrant_size = int(np.ceil(kernel_size / 2))
  samples_per_quadrant = quadrant_size**2
  img = np.pad(img, [(radius,radius),(radius,radius),(0,0)], mode="constant")
  print(img.shape)
  t = 0.01

  img_hsv = colors.rgb_to_hsv(img)
  img_gray = color.rgb2gray(img)

  quadrants_avg = np.empty((4))
  quadrant_std = np.empty((4))

  height, width, _ = img.shape
  for y in tqdm(range(radius, height-radius)):
    for x in tqdm(range(radius, width-radius), leave=False):
      # Get the indices that belong to a certain quadrant
      std = [img_hsv[y - radius: y + 1, x: x + radius + 1, 2],
            img_hsv[y - radius: y + 1, x - radius: x + 1, 2],
            img_hsv[y: y + radius + 1, x - radius: x + 1, 2],
            img_hsv[y: y + radius + 1, x: x + radius + 1, 2]]
      
      std = np.std(std, axis=(1,2))
      Q =[img[y - radius: y + 1, x: x + radius + 1],
          img[y - radius: y + 1, x - radius: x + 1],
          img[y: y + radius + 1, x - radius: x + 1],
          img[y: y + radius + 1, x: x + radius + 1]]

      min_std_index = np.argmin(std)
      avg = np.mean(Q[min_std_index], axis=(0,1))
      output[y-radius,x-radius] = avg
  
  return output

if __name__ == '__main__':
  # Setting up the input output paths
  inputDir = '../images/'
  outputDir = '../results/'
  args = parse_args()

  # main area to specify files and display blended image
  index = args.index

  # Read data and clean mask
  img = read_img(str(index).zfill(2), inputDir)

  ### The main part of the code ###

  output = kuwahara_basic(img, 5)
  output = np.clip(output, 0, 1)
  # Writing the result
  if args.outfile:
    plt.imsave(f"{outputDir}{args.outfile}.jpg", output)
  else:
    plt.imsave("{}img_{}.jpg".format(outputDir, str(index).zfill(2)), output)