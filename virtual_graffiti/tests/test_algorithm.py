import os
import cv2
import numpy as np
from django.test import TestCase
from virtual_graffiti.resources import algorithm as alg

class ColorSegmentationTest(TestCase):
    def setUp(self):
        self.frame_path = "app/static/media/comicguy.jpg"

    def test_color_segmentation(self):
        frame = cv2.imread(self.frame_path)
        red_lower = np.array([0, 100, 100])
        red_upper = np.array([10, 255, 255])
        
        segmented = alg.color_segmentation(frame, red_lower, red_upper)
        
        contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.assertTrue(len(contours) > 0)
        
        # Select the first contour as a sample contour
        sample_contour = contours[0]
        self.assertIsNotNone(sample_contour)
        self.assertIsNotNone(segmented)
        
class LaserContourTest(TestCase):
    def setUp(self):
        self.frame_path = "app/static/media/comicguy.jpg"

    def test_is_laser_contour(self):
        frame = cv2.imread(self.frame_path)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        some_lower_limit = np.array([0, 100, 100])
        some_upper_limit = np.array([255, 255, 255])
        segmented = alg.color_segmentation(frame, some_lower_limit, some_upper_limit)
        
        contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
        incorrect_contour = contours[0]
        correct_contour = contours[53]
        self.assertFalse(alg.is_laser_contour(incorrect_contour, hsv_frame))
        self.assertTrue(alg.is_laser_contour(correct_contour, hsv_frame))