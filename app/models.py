from django.db import models

class UserProfile(models.Model):
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    laser = models.CharField(max_length=10)

    class Meta:
        app_label = 'app'

class AvailableLaser(models.Model):
    laser_choices = [
        'Red',
        'Green',
        'Purple'
    ]
    
    class Meta:
        app_label = 'app'