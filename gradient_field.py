import cv2
import math
import numpy as np

class VectorField:
    
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy
        
    
    def compute_gradient(image):
        dx = cv2.Scharr(image, cv2.CV_32F, 1, 0) / 15.36
        dy = cv2.Scharr(image, cv2.CV_32F, 0, 1) / 15.36
        
        return VectorField(dx, dy)
    
    
    def direction(self, x, y):
        return math.atan2(self.dy[x, y], self.dx[x, y])
        
    def magnitude(self, x, y):
        return math.hypot(self.dy[x, y], self.dx[x, y])