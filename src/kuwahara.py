# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
import argparse

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
    
    output = img

    # Writing the result
    if args.outfile:
      plt.imsave(f"{outputDir}{args.outfile}.jpg", output)
    else:
      plt.imsave("{}img_{}.jpg".format(outputDir, str(index).zfill(2)), output)