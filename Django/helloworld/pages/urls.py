from django.urls import path
from .views import homePageView, aboutPageView, juanPageView, homePost, results

urlpatterns = [
    path('', homePageView, name='home'),
    path('about/', aboutPageView, name='about'),
    path('juan/', juanPageView, name='juan'),
    path('homePost/', homePost, name='homePost'),
    path('results/<int:choice>/<str:gmat>/', results, name='results'),
]
