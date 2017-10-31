import math
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import warp, estimate_transform
from scipy.spatial import Delaunay
from skimage import filters
from scipy import ndimage
import os

def compute_pts(arr, n_value):
    for elem in arr:
        img = plt.imread(elem)/255;
        height = len(img);
        width = len(img[0]);
        plt.imshow(img);
        img_points = plt.ginput(n=n_value, timeout=0, show_clicks=True);
        img_points += [(0, 0)];
        img_points += [(width - 1, 0)];
        img_points += [(0, height - 1)];
        img_points += [(width - 1, height - 1)];
        str = elem[:-3]+ 'txt';
        np.savetxt(fname=str, X=img_points);


def compute_avg(file1, file2, t):
    img1_array = np.loadtxt(file1);
    img2_array = np.loadtxt(file2);
    array_final = ((1 - t) * img1_array + t * img2_array)
    np.savetxt(fname= file1[:-4] + '-' + file2[:-3] + 'txt', X=array_final);


def selectTriangle(points, img1_matrix, img2_matrix, new_img, img1, img2, t):
    A = points[0];
    B = points[1];
    C = points[2];

    minx = math.floor(min(A[0], B[0], C[0]));
    maxx = math.ceil(max(A[0], B[0], C[0]));
    miny = math.floor(min(A[1], B[1], C[1]));
    maxy = math.ceil(max(A[1], B[1], C[1]));

    # affine_transform

    for x in range(minx, maxx):
        for y in range(miny, maxy):
            point = (x, y, 1);
            if (inTri(point, A, B, C)):
                product = np.matmul(img1_matrix, point);
                img1_x = int(round(product[0]));
                img1_y = int(round(product[1]));
                product = np.matmul(img2_matrix, point);
                img2_x = int(round(product[0]));
                img2_y = int(round(product[1]));
                new_img[y][x] = (1 - t) * img1[img1_y][img1_x] + (t) * img2[img2_y][img2_x];
    return new_img;


def morph(weighted_array, img1_final, img2_final, img1, img2, t):
    height = len(img1);
    width = len(img1[0]);
    new_img = np.zeros(shape=(height, width, 3));
    tri_final = Delaunay(weighted_array);
    tri_final = tri_final.simplices
    print(tri_final);
    for point in range(0, len(tri_final)):
        print(weighted_array[tri_final[point]])
        img1_matrix = computeAffine(weighted_array[tri_final[point]],
                                                     img1_final[tri_final[point]]);
        img2_matrix = computeAffine(weighted_array[tri_final[point]],
                                                     img2_final[tri_final[point]]);
        new_img = selectTriangle(weighted_array[tri_final[point]], img1_matrix, img2_matrix, new_img, img1, img2, t);
    return new_img;


def cross_prod(point1, point2, point3):
    return (point1[0] - point3[0]) * (point2[1] - point3[1]) - (point1[1] - point3[1]) * (point2[0] - point3[0])


def inTri(pt, point1, point2, point3):
    b1 = cross_prod(pt, point1, point2) < 0.0;
    b2 = cross_prod(pt, point2, point3) < 0.0;
    b3 = cross_prod(pt, point3, point1) < 0.0;
    return b1 == b2 and b2 == b3;


def create_morph(im_array, file_array):

    for elem in range(0, len(im_array) - 1):
        img1_name = im_array[elem];
        img2_name = im_array[elem + 1];
        file1, file2 = file_array[elem], file_array[elem + 1];
        img1 = plt.imread(img1_name)/255;
        img2 = plt.imread(img2_name)/255;
        folder = 'morph-sequence/'
        combo = file1[:-4] + '-' + file2[:-3] + 'txt';
        img1_final = np.loadtxt(file1);
        img2_final = np.loadtxt(file2);
        for i in range(0, 102, 5):
            compute_avg(file1, file2, i / 100.0);
            array_final = np.loadtxt(combo);
            new_img = morph(array_final, img1_final, img2_final, img1, img2, i / 100.0);
            skio.imsave(folder + str(elem*102 + i) + '.jpg', new_img);


def compute_avg_face_shape(arr):
    length = len(arr);
    first_elem = np.loadtxt(arr[0])
    arr_length = len(first_elem);
    arr_width = len(first_elem[0]);
    total_avg = np.zeros( shape = (arr_length, arr_width) );
    fname = '';
    for elem in range(0, length):
        sum_arr = np.loadtxt(arr[elem]);
        total_avg += sum_arr * (1.0/length);
    np.savetxt('avg.txt', total_avg);
    return fname;

def convert_to_avg_face_shape(img_arr, img_file_arr, avg_shape_file, t):
    ret = [];
    avg_shape = np.loadtxt(avg_shape_file);
    for i in range(0, len(img_arr)):
        img = plt.imread(img_arr[i])/255;
        img_points = np.loadtxt(img_file_arr[i]);
        new_img = morph(avg_shape, img_points, img_points, img, img, t);
        skio.imshow(new_img);
        ret += [new_img];
    return ret;

def setup_mean_face():
    person = plt.imread('person1.jpg');
    height = len(person);
    width = len(person[0]);
    for file in os.listdir("data/"):
        if file.endswith(".txt"):
            arr = np.loadtxt('data/' + file);
            arr = arr[:, 2:4];
            old_points = len(arr)
            arr[:, 0] *= width;
            arr[:, 1] *= height;
            arr = np.append(arr, [0, 0]);
            arr = np.append(arr, [[width - 1, 0]]);
            arr = np.append(arr,  [[0, height - 1]]);
            arr = np.append(arr, [[width - 1, height - 1]]);
            arr = arr.reshape(old_points + 4, 2);
            np.savetxt('data-arrays/' + file, arr)

    new_arr = [];
    for file in os.listdir("data-arrays/"):
        if(file.endswith(".txt")):
            new_arr += ["data-arrays/" + file];
    compute_avg_face_shape(new_arr);

def calc_all_danish_morphs():
    fname = 'avg.txt';
    img_arr = []
    file_arr = []
    for file in os.listdir("data/"):
        if file.endswith(".bmp"):
            img_arr += ["data/" + file];

    for file in os.listdir("data-arrays/"):
        if file.endswith(".txt"):
            file_arr += ["data-arrays/" + file]

    array = convert_to_avg_face_shape(img_arr, file_arr, fname);
    for elem in range(0, len(array)):
        plt.imsave('morphed_danes_images/' + str(elem) + '.jpg', array[elem]);


def compute_lines(img1, img2, n_value):
    source_points = [];
    source_points = np.array(source_points);
    target_points = [];
    target_points = np.array(target_points);

    img1_file_name = img1;
    img2_file_name = img2;
    img1 = plt.imread(img1);
    img2 = plt.imread(img2);

    for i in range(0, n_value):
        print(i)
        plt.imshow(img1);
        img_points = plt.ginput(n=2, timeout=0, show_clicks=True);
        source_points  = np.append(source_points, [img_points]);
        plt.imshow(img2);
        img_points = plt.ginput(n=2, timeout=0, show_clicks=True);
        target_points = np.append(target_points, [img_points]);


    str1 = img1_file_name[:-3] + 'txt';
    np.savetxt(fname=str1, X=source_points);
    str2 = img2_file_name[:-3] + 'txt';
    np.savetxt(fname=str2, X=target_points);


def computeAffine(points1, points2):
    matrix1 = np.ones([3,3])
    matrix2 = np.ones([3,3])
    matrix1[:,:-1] = points1;
    matrix2[:,:-1] = points2;
    return np.dot(matrix2.T, np.linalg.inv(matrix1.T));


spiderman = 'spiderman.jpg';
deadpool = 'deadpool.jpg';
spiderman_txt = np.loadtxt('spiderman.txt');
deadpool_txt = np.loadtxt('deadpool.txt');
spiderman_txt = spiderman_txt.reshape(28, 2)
deadpool_txt = deadpool_txt.reshape(28, 2)
spiderman_txt = np.savetxt('spiderman.txt', spiderman_txt);
deadpool_txt = np.savetxt('deadpool.txt', deadpool_txt);