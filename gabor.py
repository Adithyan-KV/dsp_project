import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def main():
    size = 100
    std = 20
    freq = 10
    orientation = 0
    gabor = generate_gabor_filter(size, std, freq, orientation)

def generate_gabor_filter(size, std, freq, orientation):
    x=np.linspace(-size/2,size/2,size)
    y=x
    x_grid, y_grid = np.meshgrid(x,y)
    x_phi = x_grid*np.cos(orientation)+y_grid*np.sin(orientation)
    y_phi = -x_grid*np.sin(orientation)+y_grid*np.cos(orientation)
    gaussian = np.exp(-(1/2)*(x_phi**2/std**2+y_phi**2/std**2))
    cosine = np.cos(2*np.pi*x_phi*freq)
    print(cosine.max(),cosine.min())
    print(gaussian.max(),gaussian.min())
    gabor = cosine*gaussian
    
    fig,plots = plt.subplots(1,3)
    fig.suptitle('Gabor filter')
    plots[0].imshow(gaussian,cmap='gray')
    plots[0].set_title(f'Gaussian with std:{std}')
    plots[1].imshow(cosine, cmap='gray')
    plots[1].set_title(f'Cosine with freq:{freq}')
    plots[2].imshow(gabor,cmap='gray')
    plots[2].set_title(f'Gabor filter with orientation:{orientation}deg')
    plt.show()

    return gabor

if __name__ == '__main__':
    main()