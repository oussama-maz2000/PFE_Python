
from rest_framework import serializers
from .models import pharmaLocation

class PharmacySerializer(serializers.ModelSerializer):
    class Meta:
        model=pharmaLocation
        fields='__all__'
                
                
                
   