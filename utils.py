import math
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import scipy.spatial
import bisect


def display_colors(colors, cols):
    """
    Displays colors in a nice format
    """
    rows = int(math.ceil(len(colors) / cols))
    box_size = 100
    
    res = np.zeros((rows * box_size, cols * box_size, 3), dtype=np.uint8)
    for y in range(rows):
        for x in range(cols):
            if y * cols + x < len(colors):
                color = [int(c) for c in colors[y * cols + x]] 
                cv2.rectangle(res, (x * box_size, y * box_size), (x * box_size + box_size, y * box_size + box_size), color, -1)
                
    return res


def display_image(image):
    """
    Displays image without having to use axis('off') everywhere
    """
    plt.axis('off')
    plt.imshow(image)
    plt.show()

    
def rgb2hsv(rgb):
    """
    Convert rgb color to hsv color
    """
    r = rgb[0] / 255
    g = rgb[1] / 255
    b = rgb[2] / 255

    cmax = max(r, g, b)
    cmin = min(r, g, b)
    
    delta = cmax - cmin
    
    # HUE CALCULATION
    if delta == 0:
        H = 0
    elif cmax == r:
        H = 60 * (((g - b) / delta) % 6)
    elif cmax == g:
        H = 60 * (((b - r) / delta) + 2)
    else:
        H = 60 * (((r - g) / delta) + 4)

    # SATURATION CALCULATION
    if cmax == 0:
        S = 0
    else:
        S = delta / cmax
        
    # VALUE CALCULATION
    V = cmax
    
    H = int(math.ceil(H))
    return (H, S, V)
    

def hsv2rgb(hsv):
    """
    Convert hsv color to rgb color
    """
    H = hsv[0]
    S = hsv[1]
    V = hsv[2]
    
    C = V * S
    
    X = C * (1 - abs(((H / 60) % 2) - 1))
    m = V - C
    
    if 0 <= H < 60:
        (R, G, B) = (C, X, 0)
    elif 60 <= H < 120:
        (R, G, B) = (X, C, 0)
    elif 120 <= H < 180:
        (R, G, B) = (0, C, X)
    elif 180 <= H < 240:
        (R, G, B) = (0, X, C)
    elif 240 <= H < 300:
        (R, G, B) = (X, 0, C)
    else:
        (R, G, B) = (C, 0, X)
        
    R = int((R + m) * 255)
    G = int((G + m) * 255)
    B = int((B + m) * 255)
    
    return (R, G, B)
    
    
def boost_colors(hsv):
    """
    Boost colors by improving brightness and saturation
    """
    H = hsv[0]
    S = hsv[1]
    V = hsv[2]
    
    r = 0.75
    S_new = np.power(S, r) + 0.05
    V_new = np.power(V, r) + 0.05

    return (H, S_new, V_new)
    
    
def color_complement(hsv):
    """
    Obtain the complement of a color within 0-180 degree by a random uniform distribution
    """
    H = hsv[0]
    S = hsv[1]
    V = hsv[2]
    
    # we want a random complement between 0 and 180 degrees
    degree = np.random.randint(0, 181)
    return (int(degree + H), S, V)


def find_closest_color(color, palette):
    """
    Compute the Euclidean distance between current color and palette
    to find the closest color
    """
    distances = []
    
    for i, val in enumerate(palette):
        distances.append([i, np.linalg.norm(color - val)])

    distances_sorted = sorted(distances, key=lambda x: x[1])
    
    color = distances_sorted[1][0]
    
    color = palette[color]

    color = [int(c) for c in color]

    return color
    
    
def find_random_color(first_color, second_color, palette):
    """
    Creates a new palette excluding two given colors, and picks a random color
    """
    new_palette = []
    
    for i in palette:
        if np.array_equal(i, first_color) or np.array_equal(i, second_color):
            pass
        else:
            new_palette.append(i)
            
    return random_color(new_palette)


def random_color(palette):
    """
    Pick a random color from the palette
    """
    rand_int = random.randint(0, len(palette) - 1)
    rand_color = palette[rand_int]
    
    # this line also makes the code work for some reason...
    rand_color = [int(c) for c in rand_color]
    
    return rand_color


def compute_color_probs(pixels, palette, k=9):
    """
    Computes the probability of colors being within a certain distance of each other
    """
    distances = scipy.spatial.distance.cdist(pixels, palette)
    maxima = np.amax(distances, axis=1)
    
    distances = maxima[:, None] - distances
    summ = np.sum(distances, 1)
    distances /= summ[:, None]
    
    distances = np.exp(k * len(palette) * distances)
    summ = np.sum(distances, 1)
    distances /= summ[:, None]
    
    return np.cumsum(distances, axis=1, dtype=np.float32)


def select_color(probs, palette):
    """
    Selects a color from the palette based on the probabilities
    """
    r = random.uniform(0, 1)
    i = bisect.bisect_left(probs, r)
    
    color = palette[i]
    color = [int(c) for c in color]
    
    return color if i < len(palette) else [int(c) for c in palette[-1]]


def random_grid(h, w, scale=1):
    """
    Creates a random grid (canvas) to paint on
    """
    r = scale // 2
    
    grid = []
    
    for i in range(0, h, scale):
        # print(i)
        for j in range(0, w, scale):
            y = random.randint(-r, r) + i
            x = random.randint(-r, r) + j
            # print((y % h, x % w))
            grid.append((y % h, x % w))

    random.shuffle(grid)
    return grid



    
    
    
    
    
    
    
