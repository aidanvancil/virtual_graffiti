from django.db import models

class Laser(models.Model):
    id = models.CharField(max_length=10, primary_key=True)
    color = models.CharField(max_length=30, default='RandomRGBHere')
    size = models.IntegerField(default=10)
    style = models.CharField(max_length=30, default='Fountain')

    class Meta:
        app_label = 'app'
        
class UserProfile(models.Model):
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    laser = models.ForeignKey(Laser, on_delete=models.SET_NULL, null=True)

    class Meta:
        app_label = 'app'        