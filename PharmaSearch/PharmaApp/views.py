import json


from pickle import FALSE, TRUE
from django.http import JsonResponse,HttpResponse,HttpRequest
import sys
sys.path.insert(0,'/Users/macbookair/Dev/project/pharma-v-recharche/PharmaSearch/PharmaApp/service')
from search import clusterPharmacy,getNearestNeigbors,setClusterInDB,getNearsetPharmacyFromDB
from django.shortcuts import render
from django.db import connection 
from rest_framework.response import Response
import asyncio
from .models import pharmaLocation
from .serializers import PharmacySerializer
from rest_framework.decorators import api_view
import pandas as pd



# Create your views here.




@api_view(['GET'])
def getData(request):
    clusterPharmacy()
    return Response('the data being grouped , please wait ')


@api_view(['POST'])
def postData(HttpRequest):
    body_unicode = HttpRequest.body.decode('utf-8')
    data=json.loads(body_unicode)
    longitude=data['longitude']
    latitude=data['latitude']
    prd=getNearestNeigbors(latitude,longitude)
    phrma=getNearsetPharmacyFromDB(prd[0])
    
    
    
    return Response(phrma)