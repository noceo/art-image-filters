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

def oil(img, kernel_size, intensity_levels):
  output = np.empty((img.shape))
  radius = kernel_size // 2
  grayscale_img = color.rgb2gray(img)
  rgb = np.empty((intensity_levels,3))
  img = np.pad(img, [(radius,radius),(radius,radius),(0,0)], mode="edge")
  grayscale_img = np.pad(grayscale_img, [(radius,radius),(radius,radius)], mode="edge")
  intensities = np.array([np.rint(pixel*intensity_levels).astype(int)-1 for pixel in grayscale_img])

  height, width, _ = img.shape
  for y in tqdm(range(radius, height-radius)):
    for x in range(radius, width-radius):
      
      intensity_Q = intensities[y-radius: y+radius+1, x-radius: x+radius+1]
      Q = img[y-radius: y+radius+1, x-radius: x+radius+1]
      rgb[rgb>0] = 0
      for j in range(intensity_Q.shape[0]):
        for i in range(intensity_Q.shape[1]):
          intensity = intensity_Q[j,i]
          rgb[intensity] += Q[j,i]
      
      unique, counts = np.unique(intensity_Q, return_counts=True)
      intensity_count = np.asarray((unique, counts)).T
      max_intensity_index = np.unravel_index(np.argmax(intensity_count), intensity_count.shape)[0]
      finalRGB = rgb[intensity_count[max_intensity_index][0]] / intensity_count[max_intensity_index][1]
      output[y-radius,x-radius] = finalRGB

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
  print(img.shape)
  
  output = oil(img, 7, 10)
  output = np.clip(output, 0, 1)
  # Writing the result
  if args.outfile:
    plt.imsave(f"{outputDir}{args.outfile}.jpg", output)
  else:
    plt.imsave("{}oil_img_{}.jpg".format(outputDir, str(index).zfill(2)), output)