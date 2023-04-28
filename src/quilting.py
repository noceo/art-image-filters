import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from skimage import color
import argparse
import cv2
import heapq
from tqdm import tqdm
import scipy

def parse_args():
  cmd = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  cmd.add_argument('-o', dest='outfile', nargs='?', help='output file')
  cmd.add_argument('-i', dest='index', type=int, default=1, help='index for finding images to read')
  cmd.add_argument('-m', dest='mode', type=str, default="random", help='quilting mode')
  cmd.add_argument('-t', dest='target', nargs='?', help='target file')
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

def get_overlap_error(patch, patch_size, overlap, result, x, y):
  error = 0
  # if there is a horizontal overlap
  if x > 0:
      error += np.sum((patch[:, :overlap] - result[y:y+patch_size, x:x+overlap])**2)
  # if there is a vertical overlap
  if y > 0:
      error += np.sum((patch[:overlap, :] - result[y:y+overlap, x:x+patch_size])**2)
  return error

def get_overlap_transfer_error(patch, target, source, patch_size, overlap, result, x, y):
  alpha = 0
  error = 0
  # first patch row of the image
  if x > 0:
      error += np.sum((patch[:, :overlap] - result[y:y+patch_size, x:x+overlap])**2)
  # first patch column of the image
  if y > 0:
      error += np.sum((patch[:overlap, :] - result[y:y+overlap, x:x+patch_size])**2)

  error = alpha * error + (1-alpha) * np.sum((source-target[y:y+patch_size, x:x+patch_size])**2)
  return error

def get_ssd_patch(texture, patch_size, overlap, tol, result, x, y, target=None, source=None):
  height, width, _ = texture.shape
  errors = np.zeros((height-patch_size, width-patch_size))
  for i in range(height - patch_size):
      for j in range(width - patch_size):
          patch = texture[i:i+patch_size, j:j+patch_size]
          if source is not None:
            source_patch = source[i:i+patch_size, j:j+patch_size]
          if target is not None:
            error = get_overlap_transfer_error(patch, target, source_patch, patch_size, overlap, result, x, y)
          else:
            error = get_overlap_error(patch, patch_size, overlap, result, x, y)
          errors[i,j] = error

  min_error = np.min(errors)
  error_indices = np.where(errors<min_error+tol)[0]
  error_index = np.random.choice(error_indices)
  i, j = np.unravel_index(error_index, errors.shape)
  return texture[i:i+patch_size, j:j+patch_size]


def get_minimum_error_cut(errors):
  error_index = [(error, [i]) for i, error in enumerate(errors[0])]
  heapq.heapify(error_index)

  height, width = errors.shape
  seen = set()

  while error_index:
    error, path = heapq.heappop(error_index)
    current_depth = len(path)
    current_index = path[-1]
    if current_depth == height:
      return path

    for delta in -1, 0, 1:
      nextIndex = current_index + delta
      if 0 <= nextIndex < width:
        if (current_depth, nextIndex) not in seen:
          cumError = error + errors[current_depth, nextIndex]
          heapq.heappush(error_index, (cumError, path + [nextIndex]))
          seen.add((current_depth, nextIndex))

                    
def get_minimum_error_patch(patch, block_size, overlap, res, x, y):
  patch = patch.copy()
  delta_y, delta_x, _ = patch.shape
  minimum_error_cut = np.zeros_like(patch, dtype=bool)

  if x > 0:
      left = patch[:, :overlap] - res[y:y+delta_y, x:x+overlap]
      left_error = np.sum(left**2, axis=2)
      for i, j in enumerate(get_minimum_error_cut(left_error)):
          minimum_error_cut[i, :j] = True

  if y > 0:
      up = patch[:overlap, :] - res[y:y+overlap, x:x+delta_x]
      up_error = np.sum(up**2, axis=2)
      for j, i in enumerate(get_minimum_error_cut(up_error.T)):
          minimum_error_cut[:i, j] = True
  np.copyto(patch, res[y:y+delta_y, x:x+delta_x], where=minimum_error_cut)

  return patch

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

def quilt_simple(texture, out_size, patch_size, tol):
  overlap = patch_size // 6
  height, width, _ = texture.shape
  patch_count_y = out_height // patch_size
  patch_count_x = out_width // patch_size
  out_height, out_width = out_size
  result = np.zeros((out_height, out_width, 3))

  for y in range(patch_count_y):
    for x in range(patch_count_x):
      j = y * patch_size
      i = x * patch_size
      patch = get_ssd_patch(texture, patch_size, overlap, tol, result, x, y)
      result[j:j+patch_size, i:i+patch_size] = patch
  
  return result

def quilt_cut(texture, out_size, patch_size, tol, target=None):
  overlap = patch_size // 6
  height, width, _ = texture.shape
  correspondence_target = None
  correspondence_texture = None
  if target is not None:
    height_target, width_target, _ = target.shape
    patch_count_y = height_target // patch_size
    patch_count_x = width_target // patch_size
    correspondence_target = scipy.ndimage.gaussian_filter(color.rgb2gray(target), 3)
    correspondence_texture = scipy.ndimage.gaussian_filter(color.rgb2gray(texture), 3)
    result = np.zeros((height_target, width_target, 3))
  else:
    patch_count_y = out_size[0] // patch_size
    patch_count_x = out_size[1] // patch_size
    out_height = out_size[0] - (patch_count_y - 1) * overlap
    out_width = out_size[1] - (patch_count_x - 1) * overlap
    result = np.zeros((out_height, out_width, 3))

  for y in tqdm(range(patch_count_y)):
    for x in tqdm(range(patch_count_x), leave=False):
      j = y * (patch_size - overlap)
      i = x * (patch_size - overlap)
      patch = get_ssd_patch(texture, patch_size, overlap, tol, result, i, j, target=correspondence_target, source=correspondence_texture)
      patch = get_minimum_error_patch(patch, patch_size, overlap, result, i, j)
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
  target = None
  if args.target:
    target = read_img(args.target, "../images/")

  ### The main part of the code ###
  if args.mode == "random":
    output = quilt_random(texture, (300,300), 50)
  elif args.mode == "simple":
    output = quilt_simple(texture, (300,300), 50, 0)
  elif args.mode == "cut":
    output = quilt_cut(texture, (300,300), 30, 0.1, target=target)
  output = np.clip(output, 0, 1)
  # Writing the result
  if args.outfile:
    plt.imsave(f"{outputDir}{args.outfile}.jpg", output)
  else:
    plt.imsave("{}quilt_{}_img_{}.jpg".format(outputDir, args.mode, str(index).zfill(2)), output)