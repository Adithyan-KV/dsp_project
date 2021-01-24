import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.filters as flt

def main():
    fingerprint = io.imread('test.tif')
    window_size = 10
    orientations = get_orientation_map(fingerprint, window_size)
    # print(np.degrees(orientations))
    
def get_orientation_map(fingerprint, window_size):
    print(np.arctan(1))
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

    # plotting the orientation map
    x_values = np.arange(0,(num_x)*window_size, window_size)
    y_values = np.arange(0,(num_y)*window_size, window_size) 
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    x_orientation, y_orientation = 10*np.cos(orientation_map), 10*np.sin(orientation_map)
    plt.imshow(fingerprint,cmap='gray')
    plt.quiver(x_grid, y_grid, x_orientation, y_orientation, color='red',scale=200, linewidth=10)
    plt.show()
    
    return orientation_map

if __name__ == '__main__':
    main()