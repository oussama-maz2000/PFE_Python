from django.urls import path

from . import views

urlpatterns = [
  path('pharmacy/', views.getData),
  path('insert/', views.postData),
  

]