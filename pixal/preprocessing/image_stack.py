import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob


parser = argparse.ArgumentParser()
parser.add_argument("images",nargs="*",help="Add visual inspection photos to upload to the production database")
args = vars(parser.parse_args())

imgar = []
for arg in args['images']:
    imgar.append(arg)
img1 = Image.open('./images/rembg_homo/transformed_22_t11_no_bg.png')
width, height = img1.size
img1 = img1.convert('HSV')
#ary2 = [[0 for _ in range(width)] for _ in range(height)]
for n, img in enumerate(imgar):
    imgo = Image.open(img) 
    hsv = imgo.convert('HSV')
   
    ary = [[0 for _ in range(width)] for _ in range(height)]
    num_cols = len(ary)
    num_rows = len(ary[0]) if num_cols > 0 else 0
    
    for i in range(num_rows):
        for p in range(num_cols):
            x, y = i, p
            h, s, v = img1.getpixel((x, y))
            ary[p][i] = ary[p][i]+v

    for i in range(num_rows):
        for p in range(num_cols):
            x, y = i, p
            h, s, v = hsv.getpixel((x, y))
            ary[p][i] = ary[p][i]+v
    print(f"Stacked image {n}")
    
plt.imshow(ary, cmap='viridis', aspect='auto')  # You can change 'viridis' to any other colormap
plt.colorbar(label='Intensity')  # Add color bar to indicate values

# Add titles and labels (optional)
plt.title('COLZ Plot2')
plt.xlabel('Column Index')
plt.ylabel('Row Index')

# Show the plot
img_name = img.split('/')[-1]
save_path = f"./images/stacked_rem/stacked_{img_name}"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.clf() 

''' 
#transformed_image_path = f'homography/transformed_{knn_ratio}_{img_name}'

plt.imshow(ary, cmap='viridis', aspect='auto')  # You can change 'viridis' to any other colormap
plt.colorbar(label='Intensity')  # Add color bar to indicate values

# Add titles and labels (optional)
plt.title('COLZ Plot2')
plt.xlabel('Column Index')
plt.ylabel('Row Index')

# Show the plot
plt.show()
plt.savefig('test2.png', dpi=300, bbox_inches='tight') 
'''