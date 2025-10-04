from django.urls import path
from . import views

app_name = 'optimization'

urlpatterns = [
    path('', views.index, name='index'),
    path('tsp/', views.solve_tsp, name='solve_tsp'),
    path('knapsack/', views.solve_knapsack, name='solve_knapsack'),
]