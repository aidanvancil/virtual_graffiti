from django.test import TestCase
from django.urls import reverse
from app.models import Laser
import json

class TestLaserSettings(TestCase):
    def setUp(self):
        self.laser = Laser.objects.create(id='Red')

    def test_set_laser_color(self):
        data = {'data': 'blue'}
        url = reverse('set_laser_color', args=[self.laser.id])
        
        response = self.client.post(url, json.dumps(data), content_type='application/json')
        
        self.assertEqual(response.status_code, 200)

        updated_laser = Laser.objects.get(id=self.laser.id)
        self.assertEqual(updated_laser.color, 'blue')

    def test_set_laser_size(self):
        data = {'data': 10}
        url = reverse('set_laser_size', args=[self.laser.id])
        
        response = self.client.post(url, json.dumps(data), content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        updated_laser = Laser.objects.get(id=self.laser.id)
        self.assertEqual(updated_laser.size, 10)

    def test_set_laser_style(self):
        data = {'data': 'dotted'}
        url = reverse('set_laser_style', args=[self.laser.id])
        
        response = self.client.post(url, json.dumps(data), content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        updated_laser = Laser.objects.get(id=self.laser.id)
        self.assertEqual(updated_laser.style, 'dotted')

    def test_get_laser(self):
        url = reverse('get_laser', args=[self.laser.id])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data['color'], self.laser.color)
        self.assertEqual(data['size'], self.laser.size)
        self.assertEqual(data['style'], self.laser.style)
