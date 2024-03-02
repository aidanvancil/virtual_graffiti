from django.db import models

'''   
    Author(s): Aidan Vancil (A), Moises Moreno (M)
    Date(s):   11/12/23 - 12/03/23
    
    Description:
    - (A) Laser Model
    - (M) User Auth Model w/ Laser
'''

class Laser(models.Model):
    id = models.CharField(max_length=10, primary_key=True)
    color = models.CharField(max_length=30, default='#777777', null=True)
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