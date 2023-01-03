"""faceRecon URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from .views import index, detect, trainer, create_dataset

urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^admins/', include('admins.urls'), name='admins'),
    url(r'^$', index),
    url(r'^create_dataset$', create_dataset),
    url(r'^detect$', detect),
    url(r'^trainer$', trainer),
]
