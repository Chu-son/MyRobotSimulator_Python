import numpy as np
import cv2

class MyParticleFilter():
    class Particle():
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.radian = 0.0
            self.weight = 0.0

    def __init__(self):
        self.particle_num = 500;
