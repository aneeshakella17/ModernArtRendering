import matplotlib.pyplot as plt
import scipy;
from scipy import ndimage;
import numpy as np
import glob

def gather_distances():
    image_list = [];
    for filename in glob.glob('rectified/*.png'):
        image_list.append(filename);
    return gather_distances_helper(image_list);

def gather_filenames():
    image_list = [];
    for filename in glob.glob('rectified/*.png'):
        image_list.append(filename);
    return image_list;

def gather_distances_helper(filenames):
    indices = [];
    for name in filenames:
        first_index = float(name[20:31]);
        second_index = float(name[32:44]);
        indices.append([first_index, second_index, 0]);
    return compute_distances(indices);

def compute_distances(indices):
    center_image_x, center_image_y, zero = indices[int(len(indices)/2)];
    correct_indices = [];
    for index in indices:
        correct_indices.append([index[0] - center_image_x, index[1] - center_image_y , 0]);
    return correct_indices;

def avg(image_list):
    height, width = np.shape(image_list[0]);
    sum_image = np.zeros(shape = (height, width));
    for image in image_list:
        sum_image += image;
    return sum_image/(len(image_list));

def shift(image, shifts):
    assert len(shifts) == len(image.shape), 'Dimensions must match'
    new = np.zeros(image.shape)
    og_selector, target_selector = [], []
    for shift in shifts:
        if shift == 0:
            og_s, target_s = slice(None), slice(None)
        elif shift > 0:
            og_s, target_s = slice(shift, None), slice(None, -shift)
        else:
            og_s, target_s = slice(None, shift), slice(-shift, None)
        og_selector.append(og_s)
        target_selector.append(target_s)
    new[og_selector] = image[target_selector]
    return new

def shift_and_avg(images, distances ,a):
    first_image = plt.imread(images[0]);
    height, width, n = np.shape(first_image);
    sum_image = np.zeros(shape = (height, width, n));

    for i in range(0, len(images)):
         new_im, dist = images[i], distances[i];
         new_dist = [dist[0], dist[1], 0];
         new_dist[0] *= a
         new_dist[1] *= a
         new_dist[0], new_dist[1] = int(new_dist[0]), int(new_dist[1]);
         new_im = plt.imread(new_im);
         shift_im = shift(new_im, new_dist);
         sum_image += shift_im;
    return sum_image/len(images);

def aperture(images, distances, a, aperture_size):
    aperture_size = 12 * aperture_size;
    new_images = images[144 - aperture_size: 145 + aperture_size];
    new_distances = distances[144 - aperture_size: 145 + aperture_size];
    return shift_and_avg(new_images, new_distances, a);

def interactive_refocus(images, distances):
    centroid_image = plt.imread(images[145]);
    plt.imshow(centroid_image);
    old_point = plt.ginput(n=1, timeout=0, show_clicks=True);
    img_point = [0,0];
    img_point[0], img_point[1] = int(old_point[0][0]), int(old_point[0][1]);
    some_images = images;
    new_distances = distances;
    height, width, n = np.shape(centroid_image);
    sum_image = np.zeros(shape = (height, width, n));
    besti, bestj = refocus(img_point, some_images[0], centroid_image);
    first_distances = new_distances[0];
    a1 = besti/first_distances[0];
    a2 = bestj/first_distances[1];
    a = 0.5 * (a1 + a2);
    for i in range(0, len(some_images)):
        x_shift = int(new_distances[i][0] * a);
        y_shift = int(new_distances[i][1] * a);
        im = plt.imread(images[i]);
        new_image_1 = np.roll(im, x_shift, 0);
        new_image_1 = np.roll(new_image_1, y_shift, 1);
        sum_image += new_image_1;
    return sum_image/len(some_images);


def refocus(img_point, image, centroid_image):
    array_of_shifts = [];
    centroid_image = centroid_image[(img_point[1] - 100): (img_point[1] + 100), (img_point[0] - 100): (img_point[0] + 100)];
    image = plt.imread(image);
    image1 = image[ (img_point[1] - 100): (img_point[1] + 100), (img_point[0] - 100): (img_point[0] + 100)];
    besti = -35;
    bestj = -35;
    best_image1 = image1;
    difference = float('inf');
    for i in range(-30,30):
        for j in range(-30,30):
            new_image_1 = np.roll(image1, i, 0);
            new_image_1 = np.roll(new_image_1, j, 1);
            new_difference = np.sum(np.sum((new_image_1 - centroid_image) ** 2));
            if(new_difference < difference):
                difference = new_difference;
                best_image1 = new_image_1;
                besti = i;
                bestj = j;
    return [besti, bestj];



def main():
    images = gather_filenames();
    distances = gather_distances();
    plt.imshow(interactive_refocus(images, distances));
    plt.show();


if __name__ == "__main__": main()
