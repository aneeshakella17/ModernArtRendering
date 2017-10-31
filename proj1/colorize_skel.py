# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images
from functools import reduce

import numpy as np
import skimage as sk
import skimage.io as skio
from skimage import transform, feature, exposure


def align(image1, image2):
    besti = -20;
    bestj = -20;
    best_image1 = image1;
    difference = float('inf');
    for i in range(-15,16):
        for j in range(-15,16):
            new_image_1 = np.roll(image1, i, 0);
            new_image_1 = np.roll(new_image_1, j, 1);
            new_difference = np.sum(np.sum((new_image_1 - image2) ** 2));
            if(new_difference < difference):
                difference = new_difference;
                best_image1 = new_image_1;
                besti = i;
                bestj = j;
    return([best_image1, besti, bestj]);



def pyramid(image):
    image_arrays = [image];
    height = np.floor(im.shape[0] / 3.0);
    height = int(height);
    while(height > 400):
        image = sk.transform.rescale(image, 0.5)
        height = np.floor(image.shape[0] / 3.0);
        height = int(height);
        image_arrays.append(image);
    return image_arrays;

def pyramid_align(image1, image2, x, y):
    besti = -6;
    bestj = -6;
    best_image1 = image1;
    difference = float('inf');
    for i in range(-5, 5):
        for j in range(-5, 5):
            new_image_1 = np.roll(image1, x + i, 0);
            new_image_1 = np.roll(new_image_1, y + j, 1);
            new_difference = np.sum((new_image_1 - image2) ** 2);
            if (new_difference < difference):
                difference = new_difference;
                best_image1 = new_image_1;
                besti = i;
                bestj = j;
    return ([best_image1, x + besti, y + bestj]);

def crop_out_borders(image):
    x_std_image = np.std(image, axis = 0);
    y_std_image = np.std(image, axis = 1);
    y_len, x_len = len(image),len(image[0]);
    border_size = int(0.2 * y_len);


    x_lower_limit = border_size;
    for i in range(border_size - 1, 1, -1):
        if(x_std_image[i][0] < 0.2 or x_std_image[i][1] < 0.2 or x_std_image[i][2] < 0.2):
            x_lower_limit = i;
            break;

    x_upper_limit = x_len - border_size;
    for j in range(x_len - border_size * 2, x_len):
        if(x_std_image[j][0] < 0.2 or x_std_image[j][1] < 0.2 or x_std_image[j][2] < 0.2):
                x_upper_limit = j;
                break;

    y_lower_limit = border_size;
    for m in range(border_size - 1, 1, -1):
        if (y_std_image[m][0] < 0.2 or y_std_image[m][1] < 0.2 or y_std_image[m][2] < 0.2):
            y_lower_limit = m;
            break;


    y_upper_limit = y_len - border_size;
    for k in range(y_len - border_size * 2, y_len):
        if(y_std_image[k][0] < 0.15 or y_std_image[k][1] < 0.15 or y_std_image[k][2] < 0.15):
                y_upper_limit = k;
                break;


    return image[x_lower_limit:x_upper_limit, y_lower_limit: y_upper_limit];


def contrast_stretching(im):
    # get image histogram
    p1 = np.percentile(im, 30);
    p2 = np.percentile(im, 70);
    img_rescale = exposure.rescale_intensity(im, in_range=(p1, p2))
    return img_rescale;

def hist_eq(im):
    return exposure.equalize_hist(im)



# name of the input file
imname = 'settlers.jpg'
# read in the c
im = skio.imread(imname)

# convert to double (might want to do this later on to save memory)
im = sk.img_as_float(im)

# compute the height of each part (just 1/3 of total)


# separate color channels


# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)


# save the image
pyramid_images = pyramid(im);
last_image = pyramid_images.pop();
height = np.floor(last_image.shape[0] / 3.0)
height = int(height);
b = last_image[:height] ;
g = last_image[height: 2 * height] ;
r = last_image[2 * height: 3 * height] ;
ag = align(g, b)
new_green = ag[0];
green_displacement_x = ag[1];
green_displacement_y = ag[2];
ar = align(r, b)
red_displacement_x = ar[1];
red_displacement_y = ar[2];
new_red = ar[0];
new_b = b;




for image in reversed(pyramid_images):
    green_displacement_x *= 2;
    green_displacement_x = int(green_displacement_x);
    green_displacement_y *= 2;
    green_displacement_y = int(green_displacement_y);
    red_displacement_x *= 2;
    red_displacement_x = int(red_displacement_x);
    red_displacement_y *= 2;
    red_displacement_y = int(red_displacement_y);
    new_height = int(np.floor(image.shape[0]/3.0));
    new_b = image[:new_height] - feature.canny(image[:new_height]);
    new_g = image[new_height: 2 * new_height] - feature.canny(image[new_height: 2 * new_height]);
    new_r = image[2 * new_height: 3 * new_height] - feature.canny(image[2 * new_height: 3 * new_height]);
    green_pastel = pyramid_align(new_g, new_b, green_displacement_x, green_displacement_y);
    red_pastel = pyramid_align(new_r, new_b, red_displacement_x, red_displacement_y);
    new_green, green_displacement_x,  green_displacement_y = green_pastel[0], green_pastel[1], green_pastel[2];
    new_red, red_displacement_x, red_displacement_y = red_pastel[0], red_pastel[1], red_pastel[2];




# create a color image

print(green_displacement_x);
print(green_displacement_y);
print(red_displacement_x);
print(red_displacement_y);


im_out = np.dstack([new_red, new_green, new_b])
fname = 'out/settlers_stretch.jpg'
# display the image

# im_out = crop_out_borders(im_out);
# im_out = im_out.astype(np.float64);
im_out = contrast_stretching(im_out);
skio.imsave(fname, im_out)
skio.imshow(im_out)
skio.show()











