import numpy as np
import pyamg
import scipy
import skimage as sk
import skimage.io as skio
from scipy.sparse import csr_matrix, lil_matrix
from skimage import filters, transform
from scipy import ndimage, sparse
from align_image_code import align_images, sharpen
import matplotlib.pyplot as plt
import colorsys
import PIL.Image

# First load images




#Pt 1.
# sharpen_image = sharpen(skio.imread('emily.jpg'))
# skio.imsave('emily-sharpen.jpg', sharpen_image);
# skio.imshow(sharpen_image);
# skio.show(
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def hybrid_image(im1, im2, sigma1, sigma2):
    [im1, im2] = align_images(im1, im2);
    im1 = sk.filters.gaussian(im1, sigma1);
    im2_aligned = im2 - (sk.filters.gaussian(im2, sigma2))
    new_image = im1/2 + im2_aligned/2;
    return new_image

def gaussian_blur(im, N):
    r = []
    g = []
    b = []
    for i in range(0, N):
        r.append(sk.filters.gaussian(im[:, :, 0], 3 * i));
        g.append(sk.filters.gaussian(im[:, :, 1], 3 * i));
        b.append(sk.filters.gaussian(im[:, :, 2], 3 * i));
    return [r, g, b];


def gaussian(im, N):
    arr = [];
    for i in range(0, N):
        arr.append(sk.filters.gaussian(im, 3 * i));
    return arr;


def laplacian(arr):
    for i in range(0, len(arr)):
        skio.imshow(arr[i] - arr[i + 1]);
        skio.show();


def mask_gaussian(im, N):
    arr = [];
    for i in range(0, N - 1):
        arr.append(sk.filters.gaussian(im, 3 * i))
    return arr;


def laplacian_blur(arr):
    r = arr[0];
    g = arr[1];
    b = arr[2];
    new_r = [];
    new_g = [];
    new_b = [];
    for i in range(0, len(r) - 1):
        new_r.append(r[i] - r[i + 1]);
        new_g.append(g[i] - g[i + 1]);
        new_b.append(b[i] - b[i + 1]);
    return [new_r, new_g, new_b];

def multiblend(source, transfer, mask):
    images = [];
    size = len(mask);
    for q in range(0, size):
        m = mask[q];
        length = len(m);
        width = len(m[0]);
        new_image = np.zeros(shape = (length, width, 3));
        for c in range(0, 3):
            s = source[c][q];
            t = transfer[c][q];
            for i in range(0, length):
                for j in range(0, width):
                    new_image[i][j][c] = m[i][j] * t[i][j] + (1 - m[i][j])*s[i][j];
        images.append(new_image);
    return images;


def toy_reconstruct(im):
 imh, imw  = np.shape(im);
 im2var = np.zeros(shape=(imh*imw, 1));
 for i in range(0, imh*imw):
    im2var[i] = int(i);

 im2var = im2var.reshape(imh, imw)
 A = np.zeros(shape = ((2*(imh)) *(imw) + 1, imh*imw));
 b = np.zeros(shape = ((2*(imh)) * (imw) + 1 , 1));

 #
 for y in range(0, imh):
    for x in range(0, imw):
       A[int(im2var[y][x])][int(im2var[y][x])] = -1;
       if(x == imw - 1 and y != imh - 1 ):
            A[int(im2var[y][x])][int(im2var[y + 1][0])] = 1;
            b[int(im2var[y][x])] = im[y + 1][0] - im[y][x];
       elif(x == imw - 1 and y == imh - 1):
           b[int(im2var[y][x])] = -im[y][x];
           continue;
       else:
            A[int(im2var[y][x])][int(im2var[y][x + 1])] = 1;
            b[int(im2var[y][x])] = im[y][x + 1] - im[y][x];


 for y in range(0, imh - 1):
    for x in range(0, imw):
        A[imh*imw + int(im2var[y][x])][int(im2var[y][x])] = -1;
        A[imh*imw + int(im2var[y][x])][int(im2var[y + 1][x])] = 1;
        b[imh*imw + int(im2var[y][x])] = im[y + 1][x] - im[y][x];



 A[2 *(imh) *(imw)][0] = 1;
 b[(2*(imw)* imh)][0] = im[0][0];
 x = sparse.linalg.lsqr(A, b)
 matrix = x[0];
 matrix = matrix.reshape(imh, imw);
 skio.imshow(matrix)
 skio.imsave("toy-result.png",matrix)
 skio.show()


def process_channel(s_channel, t_channel, im2var, mask, mixed = False):
    height = len(im2var);
    width = len(im2var[0]);
    b = np.zeros(shape =(height*width, 1));
    for h in range(0, height):
        for c in range(0, width):
            if(mask[h, c] == 1):
                sum = 4*s_channel[h, c];
                if(h + 1 != height):
                        sum -= s_channel[h + 1, c];
                if(h != 0):
                        sum -= s_channel[h - 1, c];
                if(c + 1 != width):
                        sum -= s_channel[h, c + 1];
                if(c != 0):
                        sum -= s_channel[h, c - 1];
                b[int(im2var[h, c])] = sum;
            else:
                b[int(im2var[h, c])] = t_channel[h, c];
    return b;

def mixed_process_channel(s_channel, t_channel, im2var, mask):
    height = len(im2var);
    width = len(im2var[0]);
    b = np.zeros(shape=(height * width, 1));
    for h in range(0, height):
        for c in range(0, width):
            if (mask[h, c] == 1):
                sum = 0;
                if(h + 1 != height):
                    if(abs(t_channel[h][c] - t_channel[h + 1][c]) > abs(s_channel[h][c] - s_channel[h + 1][c])):
                        sum += t_channel[h][c] - t_channel[h + 1][c];
                    else:
                        sum += s_channel[h][c] - s_channel[h + 1][c];
                if (h != 0):
                    if(abs(t_channel[h][c] - t_channel[h - 1][c]) > abs(s_channel[h][c] - s_channel[h - 1][c])):
                        sum += t_channel[h][c] - t_channel[h - 1][c];
                    else:
                        sum += s_channel[h][c] - s_channel[h - 1][c];
                if (c + 1 != width):
                    if(abs(t_channel[h][c] - t_channel[h][c + 1]) > abs(s_channel[h][c] - s_channel[h][c + 1])):
                        sum += t_channel[h][c] - t_channel[h][c + 1];
                    else:
                        sum += s_channel[h][c] - s_channel[h][c + 1];
                if (c != 0):
                    if(abs(t_channel[h][c] - t_channel[h][c - 1]) > abs(s_channel[h][c] - s_channel[h][c - 1])):
                        sum += t_channel[h][c] - t_channel[h][c - 1];
                    else:
                        sum += s_channel[h][c] - s_channel[h][c - 1];
                b[int(im2var[h, c])] = sum;
            else:
                b[int(im2var[h, c])] = t_channel[h, c];
    return b;


#
def allocate_sparse(matrix, im2var, mask):
    height = len(im2var);
    width = len(im2var[0]);
    for h in range(0, height):
        for c in range(0, width):
            if(mask[h][c] == 1):
                matrix[int(im2var[h][c]), int(im2var[h][c])] = 4;
                if(h + 1 != height):
                    matrix[int(im2var[h][c]), int(im2var[h + 1][c])] = -1;
                if(h != 0):
                    matrix[int(im2var[h][c]), int(im2var[h - 1][c])] = -1;
                if(c + 1 != width):
                    matrix[int(im2var[h][c]), int(im2var[h][c + 1])] = -1;
                if(c != 0):
                    matrix[int(im2var[h][c]), int(im2var[h][c - 1])] = -1;
            else:
                matrix[int(im2var[h][c]), int(im2var[h][c])] = 1;
    return matrix;


def poisson_blend(source, transfer, mask, best_offset, mixed = False):
    region_size = (best_offset[2] - best_offset[0], best_offset[3] - best_offset[1]);
    mask = mask[best_offset[0]:best_offset[2], best_offset[1]:best_offset[3]]
    [height, width] = np.shape(mask);
    N = height * width;
    im2var = np.zeros(shape=(N, 1));
    for i in range(0, N):
        im2var[i] = int(i);
    im2var = im2var.reshape(height, width);
    A = lil_matrix((N, N));
    A = allocate_sparse(A, im2var, mask);
    A = A.tocsr();

    source_red_channel = source[best_offset[0]: best_offset[2], best_offset[1]:best_offset[3], 0];
    source_green_channel = source[best_offset[0]:best_offset[2], best_offset[1]:best_offset[3], 1];
    source_blue_channel = source[best_offset[0]:best_offset[2], best_offset[1]:best_offset[3], 2];


    trans_red_channel = transfer[best_offset[0]: best_offset[2], best_offset[1]:best_offset[3], 0];
    trans_green_channel = transfer[best_offset[0]:best_offset[2], best_offset[1]:best_offset[3], 1]
    trans_blue_channel = transfer[best_offset[0]:best_offset[2], best_offset[1]:best_offset[3], 2];

    if(mixed):
        br = mixed_process_channel(source_red_channel, trans_red_channel, im2var, mask);
        bg = mixed_process_channel(source_green_channel, trans_green_channel, im2var, mask);
        bb = mixed_process_channel(source_blue_channel, trans_blue_channel, im2var, mask);
    else:
        br = process_channel(source_red_channel, trans_red_channel, im2var, mask);
        bg = process_channel(source_green_channel, trans_green_channel, im2var, mask);
        bb = process_channel(source_blue_channel, trans_blue_channel, im2var, mask);

    x_r = sparse.linalg.spsolve(A, br);
    x_g = sparse.linalg.spsolve(A, bg);
    x_b = sparse.linalg.spsolve(A, bb);


    x_r = x_r.reshape(height, width);
    x_g = x_g.reshape(height, width);
    x_b = x_b.reshape(height, width);

    x_r[x_r < 0] = 0;
    x_g[x_g < 0] = 0;
    x_b[x_b < 0] = 0;

    x_r[x_r > 1] = 1;
    x_g[x_g > 1] = 1;
    x_b[x_b > 1] = 1;


    transfer[best_offset[0]:best_offset[2], best_offset[1]:best_offset[3], 0] = x_r;
    transfer[best_offset[0]:best_offset[2], best_offset[1]:best_offset[3], 1] = x_g;
    transfer[best_offset[0]:best_offset[2], best_offset[1]:best_offset[3], 2] = x_b;

    return transfer;


def find_offset(mask):
   [height, width] = np.shape(mask);
   max_h = float('-inf');
   max_c = float('-inf');
   min_h = float('inf');
   min_c = float('inf');
   for h in range(0, height):
        for w in range(0, width):
            if(mask[h][w] == 1):
                min_h = min(h, min_h);
                min_c = min(w, min_c);
                max_h = max(h, max_h);
                max_c = max(w, max_c);
   return [min_h, min_c , max_h , max_c];


def color2gray(im):
    [height, width, color] = np.shape(im);
    new_im = np.zeros(shape = (height, width, 3));
    for h in range(0,height):
        for w in range(0, width):
            [h_, s, v] = colorsys.rgb_to_hsv(im[h][w][0], im[h][w][1], im[h][w][2])
            new_im[h][w][0] = h_;
            new_im[h][w][1] = s;
            new_im[h][w][2] = v;
    source = new_im[:,:,1];
    transfer = rgb2gray(im);
    mask = skio.imread('graymask.png');
    N = height * width;
    im2var = np.zeros(shape=(N, 1));
    for i in range(0, N):
        im2var[i] = int(i);
    im2var = im2var.reshape(height, width);
    A = lil_matrix((N, N));
    A = allocate_sparse(A, im2var, mask);
    A = A.tocsr();
    b = mixed_process_channel(source, transfer, mask, im2var)
    x = sparse.linalg.spsolve(A, b);
    x = x.reshape(height, width);
    # skio.imshow(x);
    # skio.show();
    return x*255;

# high sf

