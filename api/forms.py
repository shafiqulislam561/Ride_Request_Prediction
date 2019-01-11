from django import forms
#from .models import Region
from django.utils import timezone
import pandas as pd
# Import create_engine function
from sqlalchemy import create_engine
#from .forms import RegionForm


    #class Meta:
        #model = Region
        #fields =('region_name',)
        
    # Create an engine to the census database
engine_from = create_engine('mysql+pymysql://root:??&&OBHAI>>IS*({StiLl}<<AliVe)]@52.76.132.145:3306/obhai_live')

sql1="SELECT region_name FROM tb_region"
s1 = pd.read_sql_query(sql1, engine_from)

region_name=s1['region_name']


dataframe_collection = {}
# region_updated=[]
# for region in region_name :
#     #print(region)
#     sql = "SELECT hour,count( * ) total_requests,time_real,pickup_region_name FROM(SELECT session_id,HOUR(DATE_ADD(request_made_on,INTERVAL 6 HOUR)) hour,DATE_ADD(request_made_on,INTERVAL 6 HOUR) time_real,`pickup_region_name`,FLOOR(UNIX_TIMESTAMP(request_made_on)/(60*60)) time FROM `tb_engagements` WHERE `request_made_on` Between DATE_ADD(NOW() , INTERVAL -3 DAY) and NOW()  GROUP BY session_id ) c WHERE pickup_region_name LIKE '"+ region +"' GROUP BY time,pickup_region_name ORDER BY time_real,hour"
#     s = pd.read_sql_query(sql, engine_from)
#     if s.empty:
#          print('Dataframe is empty'+region)
#     else:
#         print(region)
#         region_updated.append(region)
#            # Region.objects.create(region_name=region)
#             #REGION_CHOICES[region]=region
#             #print(s)
#         dataframe_collection[region]=pd.DataFrame(s.values, columns=["hour", "total_rquests", "time_real","pickup_region_name"])

#     #regions=Region.objects.all()
# REGION_CHOICES=[tuple([region,region]) for region in region_updated]
        
# class RegionForm(forms.Form):      
#      region_name= forms.CharField(label="Please select a region", widget=forms.Select(choices=REGION_CHOICES))
     
    
    #def __init__(self, *args, **kwargs):
        #super().__init__(*args, **kwargs)
        #self.fields['region_name'].queryset=Region.objects.none()


# class UserForm(forms.Form):
#     first_name= forms.CharField(max_length=100)
#     last_name= forms.CharField(max_length=100)
#     email= forms.EmailField()
#     age= forms.IntegerField()
#     todays_date= forms.IntegerField(label="What is today's date?", widget=forms.Select(choices=INTEGER_CHOICES))

#region="Uttara (Sector-9, Sector-10, Sector-11, Abdullapur, Kamarpara, Bhatuliya,Dhour, Bamnartek, Phulbaria, Nalbhog, Noa Nagar)"
region="Banasree"
sql = "SELECT hour,count( * ) total_requests,time_real,pickup_region_name FROM(SELECT session_id,HOUR(DATE_ADD(request_made_on,INTERVAL 6 HOUR)) hour,DATE_ADD(request_made_on,INTERVAL 6 HOUR) time_real,`pickup_region_name`,FLOOR(UNIX_TIMESTAMP(request_made_on)/(60*60)) time FROM `tb_engagements` WHERE `request_made_on` Between DATE_ADD(NOW() , INTERVAL -3 WEEK) and NOW()  GROUP BY session_id ) c WHERE pickup_region_name LIKE '"+ region +"' GROUP BY time,pickup_region_name ORDER BY time_real,hour"

s = pd.read_sql_query(sql, engine_from)

#Uttara (Sector-9, Sector-10, Sector-11, Abdullapur, Kamarpara, Bhatuliya,Dhour, Bamnartek, Phulbaria, Nalbhog, Noa Nagar)

# 60 feet - Pirerbag - South Monipur
# Adabor - Muhammadia Housing
# Aftab Nagar - EWU
# Agargaon - Shere-E-Bangla Nagar - BICC
# Agargaon - Taltola - IDB - BICCpqrstuvwxyz
# Agargaon - Taltola - IDB- Radio Center

#REGION_CHOICES=[tuple([region,region]) for region in region_updated]
region_names=["Banasree","Uttara (Sector-9, Sector-10, Sector-11, Abdullapur, Kamarpara, Bhatuliya,Dhour, Bamnartek, Phulbaria, Nalbhog, Noa Nagar)","60 feet - Pirerbag - South Monipur","Adabor - Muhammadia Housing","Aftab Nagar - EWU","Agargaon - Shere-E-Bangla Nagar - BICC","Agargaon - Taltola - IDB - BICCpqrstuvwxyz","Agargaon - Taltola - IDB- Radio Center"]
data_length=["DAY","WEEK","MONTH","YEARS"]
train_ratio=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
test_ratio=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
batch_size=[1,100,200,400,500,600,700,800,900,1000]
epoch=[100,200,300,400,500,600,700,800,900,1000]
optimizer=['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
neurons = [4,100, 200, 300, 400, 500, 600, 700, 800, 900]
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
learn_rate = [0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
momentum = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
data_length_number=[1,2,3,4,5,6,7,8,9,10,11,12]
DATA_LENGTH=[tuple([data_length,data_length]) for data_length in data_length]
REGION_CHOICES=[tuple([region_names,region_names]) for region_names in region_names]        
TRAIN_RATIO=[tuple([train_ratio,train_ratio]) for train_ratio in train_ratio]
TEST_RATIO=[tuple([test_ratio,test_ratio]) for test_ratio in test_ratio]
BATCH_SIZE=[tuple([batch_size,batch_size]) for batch_size in batch_size]
EPOCH=[tuple([epoch,epoch]) for epoch in epoch]
OPTIMIZER=[tuple([optimizer,optimizer]) for optimizer in optimizer]
INIT_MODE=[tuple([init_mode,init_mode]) for init_mode in init_mode]
ACTIVATION=[tuple([activation,activation]) for activation in activation]
NEURONS=[tuple([neurons,neurons]) for neurons in neurons]
WEIGHT_CONSTRAINT=[tuple([weight_constraint,weight_constraint]) for weight_constraint in weight_constraint]
DROPOUT_RATE=[tuple([dropout_rate,dropout_rate]) for dropout_rate in dropout_rate]
LEARN_RATE=[tuple([learn_rate,learn_rate]) for learn_rate in learn_rate]
MOMENTUM=[tuple([momentum,momentum]) for momentum in momentum]
DATA_LENGTH_NUMBER=[tuple([data_length_number,data_length_number]) for data_length_number in data_length_number]
class RegionForm(forms.Form):      
     #region_name= forms.CharField(label="Please select a region", widget=forms.Select(choices=REGION_CHOICES))
     region_names=forms.MultipleChoiceField(label="Zones",widget=forms.SelectMultiple,choices=REGION_CHOICES)
     data_length=forms.CharField(label="Data length", widget=forms.Select(choices=DATA_LENGTH))
     train_ratio=forms.CharField(label="Train ratio", widget=forms.Select(choices=TRAIN_RATIO))
     batch_size=forms.CharField(label="Batch size", widget=forms.Select(choices=BATCH_SIZE))
     epoch=forms.CharField(label="Epoch", widget=forms.Select(choices=EPOCH))
     optimizer=forms.CharField(label="Optimizer", widget=forms.Select(choices=OPTIMIZER))
     init_mode=forms.CharField(label="Init mode", widget=forms.Select(choices=INIT_MODE))
     activation=forms.CharField(label="Activation", widget=forms.Select(choices=ACTIVATION))
     neurons=forms.CharField(label="Neurons", widget=forms.Select(choices=NEURONS))
     weight_constraint=forms.CharField(label="Weight constraint", widget=forms.Select(choices=WEIGHT_CONSTRAINT))
     dropout_rate=forms.CharField(label="Droup out rate", widget=forms.Select(choices=DROPOUT_RATE))
     learn_rate=forms.CharField(label="Learn rate", widget=forms.Select(choices=LEARN_RATE))
     momentum=forms.CharField(label="Momentum", widget=forms.Select(choices=MOMENTUM))
     test_ratio=forms.CharField(label="Test ratio", widget=forms.Select(choices=TEST_RATIO))
     data_length_number=forms.CharField(label="Data length number", widget=forms.Select(choices=DATA_LENGTH_NUMBER))