from colorsys import rgb_to_hsv, hsv_to_rgb
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
from skimage import feature
import scipy
from scipy import misc
from sklearn import cluster

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def intensity(rgb):

    return rgb[0] * 0.299 + 0.587 * rgb[1] + 0.114 * rgb[2];

def choose_colors(input, downsample):
    height, width, n= np.shape(input)
    blurred_input = misc.imresize(input, 0.25)
    height, width, n= np.shape(blurred_input)
    reshaped_input = np.reshape(blurred_input, (height * width, n))
    model = cluster.KMeans(n_clusters = 12);
    kmeans = model.fit_predict(reshaped_input)
    colors = model.cluster_centers_;
    colors = quick_conversion(colors);
    compliment_colors = compliments(colors)
    colors = np.vstack((colors, compliment_colors));
    new_im = np.reshape(
    colors[kmeans], (height, width, colors.shape[1]))

    return colors/255;

def quick_conversion(colors):
    new_colors = [];
    for color in colors:
        hsv = rgb_to_hsv(color[0], color[1], color[2])
        first = hsv[1] + 0.05;
        second = hsv[2] + 12.75;
        rgb = hsv_to_rgb(hsv[0], first, second);
        arr = [rgb[0], rgb[1], rgb[2]];
        new_colors.append(arr);
    return np.array(new_colors);


def compliments(colors):
    new_colors = [];
    for color in colors:
        hsv = rgb_to_hsv(color[0], color[1], color[2])
        rgb = hsv_to_rgb((hsv[0] + 0.5) % 1, hsv[1], hsv[2])
        arr = [rgb[0], rgb[1], rgb[2]];
        new_colors.append(arr);
    return np.array(new_colors);


def create_pixel_distribution(colors, im, a, h, w, std, downsample, transparency):
    blurred_input = misc.imresize(im, downsample)/255;
    height, width, n = np.shape(blurred_input)
    canvas_height, canvas_width, n = np.shape(blurred_input);
    canvas = np.ones(shape = (canvas_height * h, canvas_width * w, 4));
    x_gradient = sk.filters.sobel_h(rgb2gray(blurred_input));
    y_gradient = sk.filters.sobel_v(rgb2gray(blurred_input));
    thetas = np.arctan2(y_gradient, x_gradient);
    for y in range(0, int(height)):
        for x in range(0, int(width)):
            new_colors = best_colors(blurred_input[y][x], colors);
            I = 1 - intensity(blurred_input[y][x]);
            canvas[y * h: (y+1) * h, x * w: (x + 1) * w] = generate_cluster(new_colors, a, I, thetas[y][x], transparency, h, w, std);
    return canvas;


def generate_cluster(colors, a, I, theta, transparency, h, w, std):
     canvas = np.ones(shape = (h, w, 4))
     L, R  = 1, 2;
     A = np.array([L, R]);
     B = np.array([-1 * L, R]);
     C = np.array([L, -1 * R]);
     D = np.array([-1 * L, -1 * R]);
     angle_perturbation = np.random.randint(-400, 400) / 100;
     new_theta = np.deg2rad(theta + angle_perturbation + 90);
     rotation_matrix = np.array([[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]]);
     A, B, C, D = np.matmul(rotation_matrix, A), np.matmul(rotation_matrix, B), np.matmul(rotation_matrix,C), np.matmul(rotation_matrix,  D);
     num = int(a * I);
     prob_one = np.random.randint(90, 95, 1)/100;
     prob_second = np.random.randint(0, 7, 1)/100;
     prob_third = max(1 - prob_one - prob_second, 0);
     num_one, num_two, num_three = int(num * prob_one), int(num * prob_second), int(num * prob_third)
     num = num_one + num_two + num_three;
     num_colors = np.ones(shape = (num, 4));
     for i in range(0, num_one):
         num_colors[i, 0:3] = colors[0];
         num_colors[i, 3] = transparency;

     for i in range(0, num_two):
         num_colors[num_one + i, 0:3] = colors[1];
         num_colors[num_one + i, 3] = transparency;

     for i in range(0, num_three):
         num_colors[num_one + num_two + i, 0 :3] = colors[2];
         num_colors[num_one + num_two+ i, 3] = transparency;


     within_canvas = std * np.random.randn(num, 2) + std

     for i in range(0, num):
         point = within_canvas[i];
         center_x, center_y = point[0], point[1];

         A_x = A[0] + center_x
         B_x = B[0] + center_x
         C_x = C[0] + center_x
         D_x = D[0] + center_x

         A_y = A[1] + center_y
         B_y = B[1] + center_y
         C_y = C[1] + center_y
         D_y = D[1] + center_y

         new_A, new_B, new_C, new_D = [A_x, A_y], [B_x, B_y], [C_x, C_y], [D_x, D_y];

         min_y = int(max(min(A_y, min(B_y, min(C_y, D_y))), 0));
         max_y = int(min(max(A_y, max(B_y, max(C_y, D_y))), h));
         min_x = int(max(min(A_x, min(B_x, min(C_x, D_x))), 0));
         max_x = int(min(max(A_x, max(B_x, max(C_x, D_x))), w));


         for y in range(min_y, max_y):
             for x in range(min_x, max_x):
                 if (inRectangle(new_A, new_B, new_C, new_D, (x, y))):
                     canvas[y][x] = num_colors[i];

     return canvas;


def inRectangle(A, B, C, D, pt):
    return inTri(pt, A, B, C) or inTri(pt, D, B, C);

def inTri(pt, point1, point2, point3):
    b1 = cross_prod(pt, point1, point2) < 0.0;
    b2 = cross_prod(pt, point2, point3) < 0.0;
    b3 = cross_prod(pt, point3, point1) < 0.0;
    return b1 == b2 and b2 == b3;

def cross_prod(point1, point2, point3):
    return (point1[0] - point3[0]) * (point2[1] - point3[1]) - (point1[1] - point3[1]) * (point2[0] - point3[0])


def best_colors(pixel, colors):
    best = -1;
    second_best = -1;
    bestSSD = float('inf');

    for i in range(0, len(colors)):
        SSD = np.sum((colors[i] - pixel)**2)
        if(SSD < bestSSD):
            bestSSD = SSD;
            second_best = best;
            best = i;


    if(second_best == -1):
        bestSSD = float('inf');
        for i in range(1, len(colors)):
            SSD = np.sum((colors[i].T - pixel.T) ** 2)
            if (SSD < bestSSD):
                bestSSD = SSD;
                second_best = i;

    while(True):
        ra = np.random.randint(0, len(colors), 1)
        if(ra != best and ra!= second_best):
            new_colors = np.vstack((colors[best], colors[second_best]));
            colors = np.vstack((new_colors, colors[ra]));
            return colors

def toRGB(im):
    height, length, n = np.shape(im);
    if(n == 3):
        return im;
    new_im = np.zeros(shape = (height, length, 3));
    for y in range(0, height):
        for x in range(0, length):
            new_im[y][x][0] = im[y][x][0] * im[y][x][3];
            new_im[y][x][1] = im[y][x][1] * im[y][x][3];
            new_im[y][x][2] = im[y][x][2] * im[y][x][3];
    return new_im;


def main():
    input = plt.imread('input/up.jpg')/255;
    input = toRGB(input)
    colors = choose_colors(input, 0.2)
    new_im = create_pixel_distribution(colors, input, 200, 10, 10, 5, 0.2, 1)
    plt.imsave('output/up_pointillism.jpg', new_im)

if __name__ == "__main__": main()