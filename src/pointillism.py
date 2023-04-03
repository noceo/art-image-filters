# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from skimage import color, transform
from sklearn.cluster import KMeans
import argparse
import cv2
import random
import scipy
from tqdm import tqdm

def parse_args():
  cmd = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  cmd.add_argument('-o', dest='outfile', nargs='?', help='output file')
  cmd.add_argument('-i', dest='index', type=int, default=1, help='index for finding images to read')
  return cmd.parse_args()

# Read source, target and mask for a given id
def read_img(id, path = ""):
  img = cv2.imread(path + "img_" + id + ".jpg")
  # img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
  # info = np.iinfo(img.dtype) # get information about the image type (min max values)
  # print(info.max)
  # img = img.astype(np.float32) / info.max # normalize the image into range 0 and 1
  return img

def create_color_palette(img, k, max_size=200, n_init=10):
  height, width, _ = img.shape
  ratio = np.min([max_size/height, max_size/width])
  print("Ratio",int(height*ratio), int(width*ratio))
  height, width = int(height*ratio), int(width*ratio)
  img = cv2.resize(img, (width,height), interpolation=cv2.INTER_AREA)
  k_means = KMeans(n_clusters=k, n_init=n_init)
  k_means.fit(img.reshape(-1, 3))
  
  palette = np.rint(k_means.cluster_centers_)
  color_shifts = [(0,50,0), (15,30,0), (-15,30,0)]
  # print("Comp", palette*255)
  # print(create_color_complements(, s))
  print(np.rint(palette))
  complements = [create_color_complements(np.reshape(palette, (1,len(palette), 3)).astype(np.uint8), s) for s in color_shifts]
  complements = np.array(complements)
  complements = np.squeeze(complements)
  complements = complements.reshape((-1,3))
  # print(complements.shape, palette.shape)
  palette = np.vstack((palette, complements))
  palette = palette.astype(np.uint8)
  # print(palette)
  return palette

def create_color_complements(palette, color_shift):
  h,s,v = color_shift
  # print(palette.shape)
  hsv = cv2.cvtColor(palette, cv2.COLOR_BGR2HSV_FULL)

  if h < 0:
    h += 255
  hsv[:, :, 0] += h
  hsv[:, :, 1] = np.clip(hsv[:, :, 1] + s, 0, 255)
  hsv[:, :, 2] = np.clip(hsv[:, :, 2] + v, 0, 255)
  
  return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)
  

def show_color_palette(palette):
  rows, cols = 4, 5
  colors = len(palette)
  palette_img = np.empty((rows*80, cols*80, 3))

  for y in range(rows):
    for x in range(cols):
      color = palette[y*cols+x]
      # print(color.dtype)
      color = [c/255 for c in color]
      print(color)
      cv2.rectangle(palette_img, (x*80, y*80), (x*80+80, y*80+80), color, -1)
  
  cv2.imshow("Color Palette", palette_img)
  # cv2.waitKey(0)

def create_stroke_pos_grid(img, scale=3):
  height, width, _ = img.shape
  r = scale//2
  grid = []

  for y in range(0, height, r):
    for x in range(0, width, r):
      i = (random.randint(-r, r) + y) % height
      j = (random.randint(-r, r) + x) % width
      grid.append((i, j))

  grid = np.array(grid)
  print(grid.shape)
  np.random.shuffle(grid)
  return grid

def compute_color_probabilities(pixels, palette):
  distances = scipy.spatial.distance.cdist(pixels, palette)
  maxima = np.amax(distances, axis=1)
  distances = maxima[:, None] - distances
  distances = scipy.special.softmax(distances)
  return distances

def get_color_from_probability(probabilities, palette):
  max_prob_index = np.argsort(probabilities)[-1]
  color = palette[max_prob_index]
  return color

def create_gradient_field(img):
  grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  radius = int(round(max(grayscale_img.shape) / 50))
  s = 2*radius+1
  x = cv2.Scharr(grayscale_img, cv2.CV_32F, 1, 0) / 15.36
  y = cv2.Scharr(grayscale_img, cv2.CV_32F, 0, 1) / 15.36

  x = cv2.GaussianBlur(x, (s,s), 0)
  y = cv2.GaussianBlur(y, (s,s), 0)
  return x, y

def pointillism(img):
  palette = create_color_palette(img, 10)
  # show_color_palette(palette)
  grid = create_stroke_pos_grid(img)
  gradient_field_x, gradient_field_y = create_gradient_field(img)
  result = np.zeros(img.shape)

  randomized_pixel_colors = np.array([img[x[0],x[1]] for x in grid])
  probabilities = compute_color_probabilities(randomized_pixel_colors, palette)
  stroke_scale = int(np.ceil(np.max(img.shape) / 1000))
  for i, (y,x) in enumerate(grid):
    color = get_color_from_probability(probabilities[i], palette)
    color = [c/255 for c in color]
    gradient_y, gradient_x = (gradient_field_y[y,x], gradient_field_x[y,x])
    angle = np.degrees(np.arctan2(gradient_y, gradient_x)) + 90
    magnitude = np.linalg.norm([gradient_y, gradient_x])
    length = int(np.round(stroke_scale+stroke_scale*np.sqrt(magnitude)))
    cv2.ellipse(result, (x,y), (length,2), angle, 0, 360, color, -1, cv2.LINE_AA)

  # cv2.imshow("Result", result)
  # cv2.waitKey(0)
  result *= 255
  return result

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
  output = pointillism(img)

  # # Writing the result
  if args.outfile:
    plt.imsave(f"{outputDir}{args.outfile}.jpg", output)
  else:
    cv2.imwrite("{}pointillism_img_{}.jpg".format(outputDir, str(index).zfill(2)), output)