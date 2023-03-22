# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from skimage import color
import argparse
import cv2
from pykuwahara import kuwahara

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

def calc_quadrant(img, img_hsv, x_center, y_center, x1, x2, y1, y2, n):
  height, width, _ = img.shape
  #calc_luminance = lambda color: np.clip(np.dot(color, np.array([0.2127, 0.7152, 0.0722])), 0, 1)
  # quadrant = np.mgrid[y_center+y1:y_center+y2, x_center+x1:x_center+x2]
  # print(quadrant)
  y_start = y_center+y1
  y_end = y_center+y2
  x_start = x_center+x1
  x_end = x_center+x2
  # print(img[y_start:y_end, x_start:x_end])
  # print(img[y_start:y_end+1, x_start:x_end+1].shape)
  # avg = np.mean(img[y_start:y_end+1, x_start:x_end+1], axis=(0,1))
  # print(avg.shape)
  # print(img[y_start:y_end+1, x_start:x_end+1].shape)
  std = np.std(img_hsv[y_start:y_end+1, x_start:x_end+1, 2])
  # print(img_hsv[y_start:y_end+1, x_start:x_end+1].shape,std)
  return std


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

  i= 0
  height, width, _ = img.shape
  print(output.shape,radius, quadrant_size, samples_per_quadrant)
  for y in range(radius, height-radius):
    for x in range(radius, width-radius):
      # Get the indices that belong to a certain quadrant
      # quadrants = [img[y-radius:y+1, x-radius:x+1],
      #             img[y-radius:y+1, x:x+radius+1],
      #             img[y:y+radius+1, x-radius:x+1],
      #             img[y:y+radius+1, x:x+radius+1]]
      
      # q1 = calc_quadrant(img, img_hsv, x, y, -radius, 0, -radius, 0, samples_per_quadrant)
      # q2 = calc_quadrant(img, img_hsv, x, y, 0, radius, -radius, 0, samples_per_quadrant)
      # q3 = calc_quadrant(img, img_hsv, x, y, -radius, 0, 0, radius, samples_per_quadrant)
      # q4 = calc_quadrant(img, img_hsv, x, y, 0, radius, 0, radius, samples_per_quadrant)

      std = [img_hsv[y - radius: y + 1, x: x + radius + 1, 2],
            img_hsv[y - radius: y + 1, x - radius: x + 1, 2],
            img_hsv[y: y + radius + 1, x - radius: x + 1, 2],
            img_hsv[y: y + radius + 1, x: x + radius + 1, 2]]
      
      std = np.std(std, axis=(1,2))
      # print(std.shape)
      Q =[img[y - radius: y + 1, x: x + radius + 1],
          img[y - radius: y + 1, x - radius: x + 1],
          img[y: y + radius + 1, x - radius: x + 1],
          img[y: y + radius + 1, x: x + radius + 1]]

      # q1 = (0,0)
      # q2 = (0,0)
      # q3 = (0,0)
      # q4 = (0,0)
      min_std_index = np.argmin(std)
      # similar_std = [x for x in std if x-std[min_std_index] <= t]

      # print(quadrant_std)
      # print(min_std_index)

      avg = np.mean(Q[min_std_index], axis=(0,1))
      # print(avg)
      output[y-radius,x-radius] = avg
      # print(i)
      # print(y-radius,x-radius)
      i+=1
  print(output.shape)
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
  



  # image = cv2.imread('../images/img_01.jpg')

  # filt1 = kuwahara(image, method='mean', radius=5)
  # filt2 = kuwahara(image, method='gaussian', radius=17)    # default sigma: computed by OpenCV

  # cv2.imwrite('img_01_mean.jpg', filt1)
  # cv2.imwrite('img_01_gaus.jpg', filt2)


  output = kuwahara_basic(img, 11)
  print(output.shape)
  output = np.clip(output, 0, 1)
  # Writing the result
  if args.outfile:
    plt.imsave(f"{outputDir}{args.outfile}.jpg", output)
  else:
    plt.imsave("{}img_{}.jpg".format(outputDir, str(index).zfill(2)), output)