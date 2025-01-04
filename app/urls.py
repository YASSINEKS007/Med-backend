from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from .views import process_image

urlpatterns = [
    path("analyze-image/", process_image, name='process_image'),
    
] 