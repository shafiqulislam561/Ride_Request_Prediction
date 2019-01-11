from django.db import models
from django.shortcuts import render

# Create your views here.
#from .models import Region
from django.utils import timezone
import pandas as pd
# Import create_engine function
from sqlalchemy import create_engine
#from .forms import RegionForm




#REGION_NAME = region_updated

# class Region(models.Model):
#    # REGION_CHOICES={}
#   # Create an engine to the census database
#     engine_from = create_engine('mysql+pymysql://obhai_read:_U]igT4FU$-N/&w@13.250.35.140:3306/obhai_live')

#     sql1="SELECT region_name FROM tb_region"
#     s1 = pd.read_sql_query(sql1, engine_from)

#     region_name=s1['region_name']


#     dataframe_collection = {}
#     region_updated=[]
#     for region in region_name :
#         #print(region)
#         sql = "SELECT hour,count( * ) total_requests,time_real,pickup_region_name FROM(SELECT session_id,HOUR(DATE_ADD(request_made_on,INTERVAL 6 HOUR)) hour,DATE_ADD(request_made_on,INTERVAL 6 HOUR) time_real,`pickup_region_name`,FLOOR(UNIX_TIMESTAMP(request_made_on)/(60*60)) time FROM `tb_engagements` WHERE `request_made_on` Between DATE_ADD(NOW() , INTERVAL -8 MONTH) and NOW()  GROUP BY session_id ) c WHERE pickup_region_name LIKE '"+ region +"' GROUP BY time,pickup_region_name ORDER BY time_real,hour"
#         s = pd.read_sql_query(sql, engine_from)
#         if s.empty:
#             print('Dataframe is empty'+region)
#         else:
#             region_updated.append(region)
#            # Region.objects.create(region_name=region)
#             #REGION_CHOICES[region]=region
#         #print(s)
#     dataframe_collection[region]=pd.DataFrame(s.values, columns=["hour", "total_rquests", "time_real","pickup_region_name"])

#     #regions=Region.objects.all()
#     REGION_CHOICES=[tuple([region,region]) for region in region_updated]
  
  
  
#     region_name = models.CharField(max_length=1000000,choices=REGION_CHOICES)


