# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from skimage import color
import argparse
import cv2

def parse_args():
  cmd = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  cmd.add_argument('-o', dest='outfile', nargs='?', help='output file')
  cmd.add_argument('-i', dest='index', type=int, default=1, help='index for finding images to read')
  cmd.add_argument('-m', dest='mode', type=str, default="random", help='quilting mode')
  return cmd.parse_args()

# Read source, target and mask for a given id
def read_img(id, path = ""):
  img = plt.imread(path + "img_" + id + ".jpg")
  info = np.iinfo(img.dtype) # get information about the image type (min max values)
  img = img.astype(np.float32) / info.max # normalize the image into range 0 and 1
  return img

def get_random_patch(img, patch_size):
  height, width, _ = img.shape
  y = np.random.randint(height - patch_size)
  x = np.random.randint(width - patch_size)
  return img[y:y+patch_size, x:x+patch_size]

def quilt_random(texture, out_size, patch_size):
  height, width, _ = texture.shape
  out_height, out_width = out_size
  patch_count_y = out_height // patch_size
  patch_count_x = out_width // patch_size
  result = np.zeros((out_height, out_width, 3))

  for y in range(patch_count_y):
    for x in range(patch_count_x):
      j = y * patch_size
      i = x * patch_size
      patch = get_random_patch(texture, patch_size)
      result[j:j+patch_size, i:i+patch_size] = patch
  
  return result

if __name__ == '__main__':
  # Setting up the input output paths
  inputDir = '../images/textures/'
  outputDir = '../results/'
  args = parse_args()

  # main area to specify files and display blended image
  index = args.index

  # Read data and clean mask
  texture = read_img(str(index).zfill(2), inputDir)

  ### The main part of the code ###
  print(texture.shape)
  if args.mode == "random":
    output = quilt_random(texture, (500,500), 50)
  print(output.shape)
  output = np.clip(output, 0, 1)
  # Writing the result
  if args.outfile:
    plt.imsave(f"{outputDir}{args.outfile}.jpg", output)
  else:
    plt.imsave("{}quilt_{}_img_{}.jpg".format(outputDir, args.mode, str(index).zfill(2)), output)