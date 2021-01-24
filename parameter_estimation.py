import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.filters as flt

def main():
    fingerprint = io.imread('test.tif')
    # fingerprint = io.imread('test_2.tif') 
    fingerprint = flt.gaussian(fingerprint)
    window_size = 16
    orientations = get_orientation_map(fingerprint, window_size)
    normalized = normalize_fingerprint(fingerprint)
    x_vectors = compute_x_vectors(normalized, orientations, window_size)
    print(x_vectors)

def normalize_fingerprint(fingerprint, M_0=100, var_0 = 100):
    I = fingerprint
    M = np.mean(I)
    var = np.var(I)
    # first case
    mask_1 = I>M
    value_1 = M_0+np.sqrt((var_0*((I-M)**2))/var)
    G_1 = mask_1*value_1
    # second case
    mask_2 = I<=M
    value_2 = M_0-np.sqrt((var_0*((I-M)**2))/var)
    G_2 = mask_2*value_2

    normalized_image = G_1 + G_2
    return normalized_image

def compute_x_vectors(fingerprint, orientation_map, window_size):
    rows, columns = fingerprint.shape
    num_x = int(columns/window_size)
    num_y = int(rows/window_size)
    x_map = np.zeros((rows,columns,window_size-1))
    for i in range(num_x):
        for j in range(num_y):
            G = fingerprint
            X = np.zeros(window_size-1)
            window = fingerprint[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
            for k in range(window_size-1):
                summation = 0
                for d in range(window_size-1):
                    u = int(i+d/2*np.cos(orientation_map[i,j])+(k-d/2)*np.sin(orientation_map[i,j]))
                    v = int(j+d/2*np.sin(orientation_map[i,j])+(d/2-k)*np.cos(orientation_map[i,j]))
                    summation += G[u,v]
                X[k]=summation
            x_map[i,j]=X
    return x_map
            

def get_orientation_map(fingerprint, window_size):
    gradient_x = flt.sobel_h(fingerprint)
    gradient_y = flt.sobel_v(fingerprint)
    padding = int(window_size/2)
    rows, columns = fingerprint.shape
    num_x = int(columns/window_size)
    num_y = int(rows/window_size)
    orientation_map = np.zeros((num_x, num_y))
    for i in range(num_x):
        for j in range(num_y):
            G_x = gradient_x[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
            G_y = gradient_y[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
            numerator = np.sum(2*G_x*G_y)
            denominator = np.sum(G_x**2-G_y**2)
            if denominator == 0:
                denominator = 0.1
            theta = (1/2)*np.arctan(numerator/denominator)
            orientation_map[i,j]=theta

    # # plotting the orientation map
    # x_values = np.arange(0,(num_x)*window_size, window_size)
    # # y_values = np.arange(0,(num_y)*window_size, window_size) 
    # # x_values = np.arange(window_size,(num_x+1)*window_size, window_size)
    # y_values = np.arange(window_size,(num_y+1)*window_size, window_size) 
    # x_grid, y_grid = np.meshgrid(x_values, y_values)
    # x_orientation, y_orientation = np.sin(orientation_map), np.cos(orientation_map)
    # plt.imshow(fingerprint,cmap='gray')
    # plt.quiver(x_grid, y_grid, x_orientation, y_orientation, color='red',scale=20, linewidth=10)
    # plt.show()

    return orientation_map

if __name__ == '__main__':
    main()