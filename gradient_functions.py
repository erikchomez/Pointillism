import math
import cv2

def compute_gradient(image):
    """
    Compute the image gradient using Scharr operator
    """
    dx = cv2.Scharr(image, cv2.CV_32F, 1, 0) / 15.36
    dy = cv2.Scharr(image, cv2.CV_32F, 0, 1) / 15.36
    
    return dx, dy


def direction(dx, dy, x, y):
    """
    Compute direction of gradient
    """
    return math.atan2(dy[x, y], dx[x, y])
        
    
def magnitude(dx, dy, x, y):
    """
    Compute the magnitude of gradient
    """
    return math.hypot(dy[x, y], dx[x, y])