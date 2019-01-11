from django.conf.urls import url
from . import views

urlpatterns=[
    url(r'^neuralnets$',views.region_name,name='region_name'),
    #url(r'^neuralnets/fetch_data/$',views.fetch_data,name='fetch_data'),
    url(r'^neuralnets/preprocessdata/$',views.preprocess_data,name='preprocess_data'),
     url(r'^neuralnets/preprocessdata/predict/$',views.predict,name='predict'),
    #url(r'^neuralnets/update_region/$',views.update_region,name='update_region'),
    #url(r'^neuralnets/run_model/$',views.run_model,name='run_model'),
    #url(r'^neuralnets/select_region/$',views.select_region,name='select_region'),
]