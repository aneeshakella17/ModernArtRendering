import numpy as np
import skimage as sk
import skimage.io as skio
from scipy.sparse import csr_matrix, lil_matrix
from skimage import filters, transform
from scipy import ndimage, sparse
from align_image_code import align_images, sharpen
import matplotlib.pyplot as plt

# First load images




#Pt 1.
# sharpen_image = sharpen(skio.imread('emily.jpg'))
# skio.imsave('emily-sharpen.jpg', sharpen_image);
# skio.imshow(sharpen_image);
# skio.show();
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def hybrid_image(im1, im2, sigma1, sigma2):
    im1_aligned, im2_aligned = align_images(im1, im2)
    low_pass_im1 = sk.filters.gaussian(im1_aligned, sigma1);
    im2_aligned = (im2_aligned)/2 - (sk.filters.gaussian(im2_aligned, sigma2))/2
    return rgb2gray((low_pass_im1)/2 + (im2_aligned)/2)

def gaussian_blur(im, N):
    r = []
    g = []
    b = []
    for i in range(0, N):
        r.append(sk.filters.gaussian(im[:, :, 0], 3 * i));
        g.append(sk.filters.gaussian(im[:, :, 1], 3 * i));
        b.append(sk.filters.gaussian(im[:, :, 2], 3 * i));
    return [r, g, b];


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
        new_image = np((length, width, 3));
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
 print(np.shape(matrix))
 skio.imshow(matrix)
 skio.show()





def poisson_blend(s, t, m):
    # height = len(m);
    # width = len(m[0]);
    # new_image = np.zeros((height, width, 3));
    # for c in range(0, 3):
    #     for h in range(0, height):
    #         for w in range(0, width):
    #             new_image[h][w][c] = m[h][w]*s[h][w][c] + (1 - m[h][w]) * t[h][w][c];
    new_image = skio.imread('current_blend.png')
    return poisson_color_process(new_image)



def poisson_color_process(new_image):
    height = len(new_image);
    width = len(new_image[0]);

    red_channel = new_image[:, :, 0];
    green_channel = new_image[:, :,1];
    blue_channel = new_image[:, :, 2];



    im2var = np.zeros(shape=(height * width, 1));
    for i in range(0, height * width):
        im2var[i] = int(i);
    im2var = im2var.reshape(height, width);

    print("Channels Processing")
    br = process_channel(red_channel, im2var);
    bg = process_channel(green_channel, im2var);
    bb = process_channel(blue_channel, im2var);



    print('Populate')
    N = height * width;
    A = lil_matrix((N, N));
    A = allocate_sparse(A, im2var);


    print("Solving")
    A = sparse.linalg.tocsr(A);
    print("x_r");
    x_r = sparse.linalg.spsolve(A, br);
    print("x_g")
    x_g = sparse.linalg.spsolve(A, bg);
    print("x_b")
    x_b = sparse.linalg.spsolve(A, bb);
    new_image = np.zeros(shape = (height, width, 3));
    x_r = x_r.reshape(height, width);
    x_g = x_g.reshape(height, width);
    x_b = x_b.reshape(height, width);
    new_image[:, :, 0] = x_r;
    new_image[:, :, 1] = x_g;
    new_image[:, :, 2] = x_b;
    return new_image;


def process_channel(channel, im2var):
    height = len(im2var);
    width = len(im2var[0]);
    b = np.zeros(shape =(height*width, 1));
    for h in range(0, height):
        for c in range(0, width):
            sum = 4*channel[h][c];
            if(h + 1 != height ):
                sum -= channel[h + 1][c];
            if(h != 0):
                sum -= channel[h - 1][c];
            if(c + 1 != width):
                sum -= channel[h][c + 1];
            if(c != 0):
                sum -= channel[h][c - 1];
            b[int(im2var[h][c])] = sum;
    return b;


def allocate_sparse(matrix, im2var):
    height = len(im2var);
    width = len(im2var[0]);
    for h in range(0, height):
        for c in range(0, width):
            matrix[int(im2var[h][c]), int(im2var[h][c])] = 4;
            if(h + 1 != height):
                matrix[int(im2var[h][c]), int(im2var[h + 1][c])] = -1;
            if(h != 0):
                matrix[int(im2var[h][c]), int(im2var[h - 1][c])] = -1;
            if(c + 1 != width):
                matrix[int(im2var[h][c]), int(im2var[h][c + 1])] = -1;
            if(c != 0):
                matrix[int(im2var[h][c]), int(im2var[h][c - 1])] = -1;
    return matrix;










# high sf
apple = plt.imread('apple.jpeg')/255
# low sf
orange = plt.imread('orange.jpeg')/255
im = [[1, 2, 3], [4, 5, 6]];
toy = plt.imread('toy_problem.png')
# toy_reconstruct(toy)


# Next align images (this code is provided, but may be improved)
## You will provide the code below. Sigma1 and sigma2 are arbitrary
## cutoff values for the high and low frequecies
# blended_image = hybrid_image(im1, im2, 6, 3);
# skio.imshow((blended_image));
# skio.show();
# skio.imsave("blend.jpg",blended_image)


## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function

# N = 6 # suggested number of pyramid levels (your choice)
# gauss_apple = gaussian_blur(apple, N);
# gauss_orange = gaussian_blur(orange, N);
# mask = plt.imread('mask2.jpg')/255;
# mask = rgb2gray(mask);
# mask[mask > 0.5] = 1;
# mask[mask < 0.5] = 0;
# laplacian_apple = laplacian_blur(gauss_apple);
# laplacian_orange = laplacian_blur(gauss_orange);
# gauss_mask = mask_gaussian(mask, N)
# laplace_blend = multiblend(laplacian_apple, laplacian_orange, gauss_mask)
# gaussian_blend = multiblend(gauss_apple, gauss_orange, gauss_mask);
# sum = gaussian_blend[4] + laplace_blend[3] + laplace_blend[2] + laplace_blend[1] + laplace_blend[0];
# skio.imshow(sum)
# skio.show()
mask = plt.imread('combined.png');
transfer = plt.imread('im3.jpg')/255;
source = plt.imread('combinedpenguinartic.png');
mask = rgb2gray(mask);
img = poisson_blend(source, transfer, mask)
skio.imshow(img);
skio.show();

#tot_image = blended_images[0]/N
#for q in range(1, N):
#     tot_image += blended_images[q]/N;
# skio.imshow(tot_image);
# skio.show();