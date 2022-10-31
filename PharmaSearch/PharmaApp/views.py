from operator import index
from pickle import FALSE, TRUE
from django.http import JsonResponse,HttpResponse
import sys
sys.path.insert(0,'/Users/macbookair/Dev/project/pharma-v-recharche/PharmaSearch/PharmaApp/service')
from search import clusterPharmacy,getNearestNeigbors
from django.shortcuts import render
from django.db import connection 
from rest_framework.response import Response
import asyncio
from .models import pharmaLocation
from .serializers import PharmacySerializer
from rest_framework.decorators import api_view
import pandas as pd
from time import sleep

import httpx
# Create your views here.




@api_view(['GET'])
def getData(request):
    pharmacy=pharmaLocation.objects.all()
    pharmacy_serializer=PharmacySerializer(pharmacy,many=TRUE)
    dataframe=pd.read_sql_query("select * from PharmaApp_pharmalocation",connection)
    df,client= clusterPharmacy(dataframe)
    predection=getNearestNeigbors(df,client)
    print(predection)    
    return Response(pharmacy_serializer.data)


@api_view(['POST'])
def postData(request):
    serializer=PharmacySerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
    return Response(serializer.data)