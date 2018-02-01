import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
from skimage import feature
from skimage.filters import sobel

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def create_brush_strokes(L, R, im):
    height, width, n = np.shape(im);
    number_of_brush_strokes = int(height * width * 2/(L * R));
    return [(np.random.randint(0, height), np.random.randint(0, width)) for k in range(number_of_brush_strokes)]

def color_bursh_stroke(L, R, brush_centers, im, lines):
    height, width, n = np.shape(im);
    new_im = np.zeros(shape = (height, width, n));
    x_gradient = sk.filters.sobel_h(rgb2gray(im));
    y_gradient = sk.filters.sobel_v(rgb2gray(im));
    thetas = np.arctan2(y_gradient, x_gradient);

    for center in brush_centers:
        center_y = center[0];
        center_x = center[1];
        color = np.array([0, 0, 0]);
        angle_perturbation = np.random.randint(-400, 400)/100;
        theta = np.rad2deg(thetas[center_y][center_x]);
        new_theta = np.deg2rad(theta + angle_perturbation + 90);
        rotation_matrix = np.array([[np.cos(new_theta) , -1*np.sin(new_theta)],[np.sin(new_theta), np.cos(new_theta)]]);
        A = np.array([L, R]);
        B = np.array([-1 * L, R]);
        C = np.array([L, -1 * R]);
        D = np.array([-1 * L, -1 * R]);
        A, B, C, D = np.matmul(rotation_matrix, A), np.matmul(rotation_matrix, B), np.matmul(rotation_matrix, C), np.matmul(rotation_matrix, D)

        A[0] += center_x
        B[0] += center_x
        C[0] += center_x
        D[0] += center_x

        A[1] += center_y
        B[1] += center_y
        C[1] += center_y
        D[1] += center_y

        min_y = int(max(min(A[1], min(B[1], min(C[1], D[1]))), 0))  ;
        max_y = int(min(max(A[1], max(B[1], max(C[1], D[1]))), height))  ;
        min_x = int(max(min(A[0], min(B[0], min(C[0], D[0]))), 0));
        max_x = int(min(max(A[0], max(B[0], max(C[0], D[0]))), width)) ;

        count = 0;
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                if(inRectangle(A, B, C, D, (x, y))):
                    color = np.add(color, im[y][x]);
                    count += 1;

        avg_color = color/count;

        for i in range(0, 3):
            perturbation = np.random.randint(-400, 400);
            perturbation = perturbation/10000;
            avg_color[i] = max(min(avg_color[i] + perturbation, 1), 0);

        A, B, C, D = clipRectangle(A,B,C,D,min_y, min_x, max_y, max_x, (center_x, center_y), lines);

        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                if(inRectangle(A, B, C, D, (x, y))):
                    new_im[y][x] = avg_color;


    return new_im;


def clipRectangle(A, B, C, D, min_y, min_x, max_y, max_x, center, lines):
    first_set = False;
    first_white_pixel = [];
    last_white_pixel = [];
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            if (inRectangle(A, B, C, D, (x, y))):
                if (lines[y][x] == 1 and not first_set):
                    first_white_pixel = (x, y);
                    first_set = True
                elif (lines[y][x] == 1 and first_set):
                    last_white_pixel = (x, y);


    if(len(first_white_pixel) == 0 or len(last_white_pixel) == 0):
        return A,B,C,D
    elif(inRectangle(A, B, first_white_pixel, last_white_pixel, center)):
        return A, B, first_white_pixel, last_white_pixel;
    elif(inRectangle(A, C, first_white_pixel, last_white_pixel, center)):
        return C, D, first_white_pixel, last_white_pixel;
    elif(inRectangle(A, D, first_white_pixel, last_white_pixel, center)):
        return A, D, first_white_pixel, last_white_pixel;
    elif(inRectangle(B, C, first_white_pixel, last_white_pixel, center)):
        return B, C, first_white_pixel, last_white_pixel;
    elif(inRectangle(B, D, first_white_pixel, last_white_pixel, center)):
        return B, D, first_white_pixel, last_white_pixel;
    elif(inRectangle(C, D, first_white_pixel, last_white_pixel, center)):
        return C, D, first_white_pixel, last_white_pixel;


def inRectangle(A, B, C, D, pt):
    return inTri(pt, A, B, C) or inTri(pt, D, B, C);

def inTri(pt, point1, point2, point3):
    b1 = cross_prod(pt, point1, point2) < 0.0;
    b2 = cross_prod(pt, point2, point3) < 0.0;
    b3 = cross_prod(pt, point3, point1) < 0.0;
    return b1 == b2 and b2 == b3;

def cross_prod(point1, point2, point3):
    return (point1[0] - point3[0]) * (point2[1] - point3[1]) - (point1[1] - point3[1]) * (point2[0] - point3[0])


def construct_edges(input):
    edges = feature.canny(rgb2gray(input), 3);
    new_im = np.zeros(shape = (len(edges), len(edges[0])));
    for y in range(0, len(edges)):
        for x in range(0, len(edges[0])):
            if(edges[y][x] > 0.05):
                new_im[y][x] = 1;
            else:
                new_im[y][x] = 0;
    return new_im;

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
    input = toRGB(input);
    lines = construct_edges(input);
    centers = create_brush_strokes(5, 2, input);
    new_im = color_bursh_stroke(5, 2, centers, input, lines);
    plt.imsave('output/up_impression.jpg', new_im);

if __name__ == "__main__": main()
