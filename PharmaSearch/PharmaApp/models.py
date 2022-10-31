from email.policy import default
from operator import mod
from django.db import models


# Create your models here.
class pharmaLocation(models.Model):
    pharmaName = models.CharField( max_length=255, blank=True, null=True)
    latitude = models.FloatField()
    longitude = models.FloatField()
    cluster=models.IntegerField(null=True)
   
class Meta:
    db_table = "pharmaLocation"
    
    
