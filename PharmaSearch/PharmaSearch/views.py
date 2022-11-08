from rest_framework.response import Response
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import HttpRequest
import json
from .search import getNearestNeigborsClient,setClusterInDB,getNearsetPharmacyFromDB,clusterPharmacy

@api_view(['GET'])
def welcome(HttpRequest):
    return Response("hello world from django ")


@api_view(['POST'])
def getNearestPharmacy(HttpRequest):
    body_unicode = HttpRequest.body.decode('utf-8')
    data=json.loads(body_unicode)
    prd=getNearestNeigborsClient(data['latitude'],data['longitude'])
    return Response(prd[0]) 