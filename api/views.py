from django.shortcuts import render
from sklearn.model_selection import GridSearchCV
from numpy.testing import assert_allclose
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
# keras library import  for Saving and loading model and weights
from keras.models import model_from_json
import os.path
import matplotlib.pyplot as plt
# Create your views here.
#from .models import Region
from django.utils import timezone
import pandas as pd
# Import create_engine function
from sqlalchemy import create_engine
from .forms import RegionForm
from keras.wrappers.scikit_learn import KerasClassifier

import keras
import pandas as pd
import numpy as np
#from keras.layers.core import Dense, Activation, Dropout
import sklearn
import math
import matplotlib
import numpy

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
#from pandas import datetime
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from math import sqrt
#from matplotlib import pyplot
from numpy import array




from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import LSTM
from pandas import read_csv
#from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

from .forms import RegionForm

import json
import simplejson

# import keras
# import pandas as pd
# import numpy as np
# from keras.layers.core import Dense, Activation, Dropout
# import sklearn



#global df,n_lag,n_seq,n_in,n_out,n_test,n_epochs,n_batch,n_neurons

# n_lag = 24
# n_seq = 24
# n_in=24
# n_out=24
# n_test = 24
# n_epochs = 1500
# n_batch = 200
# n_neurons = 400

def region_name(request):
  
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
#             print('Dataframe is empty')
#         else:
#             region_updated.append(region)
#             Region.objects.create(region_name=region)
#         #print(s)
#     dataframe_collection[region]=pd.DataFrame(s.values, columns=["hour", "total_rquests", "time_real","pickup_region_name"])

#     regions=Region.objects.all()
    #regions=Region.objects.create()
    form=RegionForm
    return render(request, 'api/region_name.html', {'form':form,})

def preprocess_data(request):
    if request.method =="POST":
        region=RegionForm(request.POST)
        #if region.is_valid():
        #post=form.save(commit=False)
        #post.author=request.user
        #post.published_date=timezone.now()
        if region.is_valid():
            global region_variable
            #region_variable=region.cleaned_data['region_name']
            #region_variable
            global data_length
            global train_ratio
            global batch_size
            global epoch
            global optimizer
            global init_mode
            global activation
            global neurons
            global weight_constraint
            global dropout_rate
            global learn_rate
            global momentum
            global test_ratio
            global data_length_number
            
            
            
            region_variable=region.cleaned_data['region_names']
            data_length=region.cleaned_data['data_length']
            train_ratio=region.cleaned_data['train_ratio']
            batch_size=region.cleaned_data['batch_size']
            epoch=region.cleaned_data['epoch']
            optimizer=region.cleaned_data['optimizer']
            init_mode=region.cleaned_data['init_mode']
            activation=region.cleaned_data['activation']
            neurons=region.cleaned_data['neurons']
            weight_constraint=region.cleaned_data['weight_constraint']
            dropout_rate=region.cleaned_data['dropout_rate']
            learn_rate=region.cleaned_data['learn_rate']
            momentum=region.cleaned_data['momentum']
            test_ratio=region.cleaned_data['test_ratio']
            data_length_number=region.cleaned_data['data_length_number']
            print(region_variable)
            # engine_from = create_engine('mysql+pymysql://root:tRB9zwTeFiPPAOYz@52.76.132.145:3306/obhai_live')
            # sql = "SELECT hour,count( * ) total_requests,time_real FROM(SELECT session_id,HOUR(DATE_ADD(request_made_on,INTERVAL 6 HOUR)) hour,DATE_ADD(request_made_on,INTERVAL 6 HOUR) time_real,`pickup_region_name`,FLOOR(UNIX_TIMESTAMP(request_made_on)/(60*60)) time FROM `tb_engagements` WHERE `request_made_on` Between DATE_ADD(NOW() , INTERVAL -1 MONTH) and NOW()  GROUP BY session_id ) c WHERE pickup_region_name LIKE '"+ region_variable +"' GROUP BY time,pickup_region_name ORDER BY time_real,hour"
            # dataset = pd.read_sql_query(sql, engine_from)
            
            #Updated
            newlist = []
            date=[]
            time=[]
            day=[]
            #
            #list1=dataset['date']
            #list1=dataset['time_real']
            #date=list1
            #print(list1)
            #print(date)
            #print(dataset)
            global dataframe_collection_new
            dataframe_collection_new={}
            for region in region_variable:
                
                newlist = []
                date=[]
                time=[]
                day=[]

                print(region)
                engine_from = create_engine('mysql+pymysql://root:??&&OBHAI>>IS*({StiLl}<<AliVe)]@52.76.132.145:3306/obhai_live')
                sql = "SELECT hour,count( * ) total_requests,time_real FROM(SELECT session_id,HOUR(DATE_ADD(request_made_on,INTERVAL 6 HOUR)) hour,DATE_ADD(request_made_on,INTERVAL 6 HOUR) time_real,`pickup_region_name`,FLOOR(UNIX_TIMESTAMP(request_made_on)/(60*60)) time FROM `tb_engagements` WHERE `request_made_on` Between DATE_ADD(NOW() , INTERVAL -1 WEEK) and NOW()  GROUP BY session_id ) c WHERE pickup_region_name LIKE '"+region+"' GROUP BY time,pickup_region_name ORDER BY time_real,hour"
                dataset = pd.read_sql_query(sql, engine_from)
                list1=dataset['time_real']
                for line in list1:
                    line_list = line.date()
                    date.append(line_list)
                    #time.append(line_list[1])
    
                values=dataset.values
                #global df  
                new_dataset = DataFrame()
                df = DataFrame()
                df_date=DataFrame( date,columns=['date'])
                df_hour=DataFrame( values[:,0],columns=['hour'])
                df_orginal_total_requets=DataFrame( values[:,1],columns=['original_total_requests'])
                #df_pickup_region_name=DataFrame( values[:,3],columns=['pickup_region_name'])
                df=pd.concat([ df_hour,df_orginal_total_requets,df_date],axis=1)

                array=sorted(set(date))

                new_dataset = pd.DataFrame(columns=['original_total_requests', 'hour','date'])
                #print(array[0])

                i=0
                flag=False
                for a in array:
                    for h in range(0,24):
                        flag=False
                        for row in range(0,len(df)):
                            if a==df['date'][row] and df['hour'][row]==h:
                                #if df['original_total_requests'][row]>10:
                                    #print("inside if")
                                    #new_dataset.loc[i]=[10,h,df['date'][row]]
                                    #print(df['original_total_requests'][row])
                                    # i+=1
                                    # flag=True
                                    # break
                                #else :
                                    #print("inside else")
                                new_dataset.loc[i]=[df['original_total_requests'][row],h,df['date'][row]]
                                i+=1
                                flag=True
                                break       
                        if flag==False:
                            new_dataset.loc[i]=[0,h,a]
                            i+=1
     
                mylist=new_dataset['date']
                format = "%Y-%m-%d"
                #t.strftime('%m/%d/%Y')
                i=0
                for line1 in mylist:
                    date1=line1.strftime('%Y-%m-%d')
                    newlist.append([datetime.datetime.strptime(date1,format).strftime('%A')])
                    day.append(newlist[i][0])
                    i=i+1

                values=new_dataset.values
                df_hour=DataFrame( values[:,1],columns=['hour'])
                df_orginal_total_requets=DataFrame( values[:,0],columns=['original_total_requests'])
                df_date=DataFrame( values[:,2],columns=['date'])
                df_day=DataFrame( day,columns=['day'])
                #df_pickup_region_name=DataFrame( values[:,3],columns=['pickup_region_name'])
                df=pd.concat([df_orginal_total_requets,df_hour,df_day,df_date],axis=1)
                dataframe_collection_new[region]=pd.DataFrame(df.values, columns=["original_total_requests","hour", "day","date"])
                #dataframe_collection_new[region]=pd.DataFrame(df.values, columns=["original_total_requests","hour", "day","date","pickup_region_name"])
                #headers=df.dtypes.index
                #df=df.set_index('date')
                #df=df.sort_values('date')
                #print(headers)
                print(df)
                #return render(request,'api/preprocessing_done.html',{'df':df})
            
            #post.save()
            #return redirect('post_detail',pk=post.pk)

        else:
            form=RegionForm   
            return render(request,'api/region_name.html',{'form':form})
    #return render(request,'api/preprocessing_done.html',{'df':df})
        return render(request,'api/preprocessing_done.html',{'df':df}) 
#def train_model(request):




# convert time series into supervised learning problem
def series_to_supervised(data, n_in, n_out, dropnan=True):
    n_vars = 24 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg



# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)



# transform series into train and test sets for supervised learning
def prepare_data(values, n_test, n_in,n_out, n_seq):
    # extract raw values
    raw_values = values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_in, n_out)
    supervised_values = supervised.values
    print("Supervised values")
    print(supervised_values.shape)
    # split into train and test sets
    #train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    #print(train)
    train = supervised_values
    #train=supervised_values[0:-n_test]
    #print(train.shape,test.shape)
    print("Train shape")
    print(train.shape)
    
    return scaler, train




# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    print("X shape")
    print(X.shape)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    print(X.shape)
    # design network
    model = Sequential()
    #model.add(LSTM(batch_input_shape=(n_batch, X.shape[1], X.shape[2]),output_dim=1,return_sequences=True,stateful=True))
    model.add(LSTM(input_shape=(X.shape[1], X.shape[2]),output_dim=24,return_sequences = True))#Updated
    model.add(LSTM(n_neurons))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adamax')
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2, shuffle=False)
        model.reset_states()
    return model




# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    #print(forecast)
    # convert to array
    #return [x for x in forecast[0, :]]
    return [x for x in forecast]


# evaluate the persistence model
def make_forecasts(model, n_batch, test, n_lag):
    forecasts = list()
    #for i in range(len(test)):
        #X, y = test[i, 0:n_lag], test[i, n_lag:]
    X = test
    # make forecast
    forecast = forecast_lstm(model, X, n_batch)
    # store the forecast
    forecasts.append(forecast)
    return forecasts




# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted
 

# inverse data transform on forecasts
def inverse_transform(values, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(values) - n_test + i - 1
        last_ob = values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted




# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        #actual = [row[i] for row in test]
        actual = test
        #predicted = [forecast[i] for forecast in forecasts]
        predicted = forecasts
        rmse = sqrt(mean_squared_error(actual[i], predicted[i]))
        print('t+%d RMSE: %f' % ((i+1), rmse))



# plot the forecasts in the context of the original dataset
def plot_forecasts(values, forecasts, n_test):
    # plot the entire dataset in blue
    pyplot.plot(values)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = len(values) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [values[off_s]] + forecasts[i]
        pyplot.plot(xaxis, yaxis, color='red')
    # show the plot
    pyplot.show()
#from keras.optimizers import Adamax
# Function to create model, required for KerasClassifier
def create_model(look_back,neurons=1,optimizer='adam',init_mode='uniform',activation='relu',weight_constraint=0,dropout_rate=0.0):
	# create model
	# model = Sequential()
	# model.add(Dense(12, input_dim=8, activation='relu'))
	
    # model.add(Dense(1, activation='sigmoid'))
	# # Compile model
	# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
    model = Sequential()
    
    #X, y = train[:, 0:n_lag], train[:, n_lag:]#Updated
    #print("X shape")
    #print(X.shape)
    #X = X.reshape(X.shape[0], 1, X.shape[1])#Updated
    #print(X.shape)
    #model.add(LSTM(input_shape=(X.shape[1], X.shape[2]),output_dim=24,return_sequences = True))#Updated
    #model.add(LSTM(int(float(neurons)),input_shape=(1, look_back)))#Updated
    #print("The hyper parameters :"+"neurons : "+neurons+"optimizer : "+optimizer+"init_mode :"+init_mode+"activation :"+activation+" weightconstraint :"+weight_constraint+"drop out rate :"+dropout_rate)
    #model.add(Dropout(dropout_rate))
    
    #%%%model.add(LSTM(int(float(neurons)),input_shape=(1, look_back),kernel_initializer=init_mode,activation=activation))
    #%%%model.add(Dropout(dropout_rate))
    #model.add(Dense(y.shape[1],activation='relu'))
    #%%%model.add(Dense(1,activation='relu'))
    #optimizer = Adamax(lr=learn_rate)
    #%%%model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['accuracy'])
    

    model = Sequential()
    model.add(LSTM(int(float(neurons)), input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adamax')
    #model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)



    return model
    # convert an array of values into a dataset matrix


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


def predict(request):
    # load dataset
    #series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    #series = read_csv("C:/Users/HP/Desktop/test_v3_v1.csv")
    #series.drop('index', axis=1, inplace=True)
    #global df_orginal_total_requets
    results_collection={}
    test_results_collection={}
    for key in dataframe_collection_new:
        print(key)
        global df_orginal_total_requets
        df_orginal_total_requets=DataFrame()
        #global df_orginal_total_requets
        #df=dataframe_collection_new[key]
        df=dataframe_collection_new[key]
        print(df.head(5))
        # save to file
        #series.to_csv('predict_requests.csv')
        #print(series)
        #series = read_csv('predict_requests.csv', header=0, index_col=4)
        #series.drop('Unnamed: 0', axis=1, inplace=True)

        print(df)
        values = df.values

        df_orginal_total_requets=DataFrame( values[:,0],columns=['original_total_requests'])

        #values=series.values
        #encoder = LabelEncoder()
        #array2=encoder.fit_transform(values[:,2])
        #array2=encoder.inverse_transform(array2)
        #values[:,2] = encoder.fit_transform(values[:,2])
        #print(series)
        # summarize first 5 rows

        #Updated
        #values = values.astype('float32')
        values1=values[:,0]
        #values2=values1
        #values1=values1.astype('float32')
        #values2=values2.astype('float32')
        #values1=values[:,0]
        #values=values1[0:2519]
        # values=values1[0:-n_test]
        #

        # print(values)
        # print(values.shape)


        #Updated
        # configure
        #global n_lag
        #n_lag = 72
        #n_seq = 72
        #n_in=72
        #n_out=72
        #n_test = 73
        #n_epochs = 1500
        #n_batch = 200
        #n_neurons = 400
        #
        #Updated
        #values=values1[0:-n_test]

        #print(values)
        #print("Values shape")
        #print(values.shape)
        #
        
        #Updated
        # prepare data
        global train
        #scaler, train = prepare_data(values, n_test, n_in,n_out, n_seq)
        #

        #Updated
        #X, y = train[:, 0:n_lag], train[:, n_lag:]
        #print("X shape")
        #print(X.shape)
        #X = X.reshape(X.shape[0], 1, X.shape[1])
        #print(X.shape)
        # create model
        #model = KerasClassifier(build_fn=create_model, verbose=2)
        # define the grid search parameters
        #batch_size = [200]
        #epochs = [180]
        #optimizer = ['Adamax']
        #init_mode = ['glorot_normal']
        #activation = ['relu']
        #neurons = [100]
        #weight_constraint = [0]
        #dropout_rate = [0.9]
        #learn_rate = [0.2]
        #momentum = [ 0.4]
        #
        
        # batch_size = [10, 20, 40, 60, 80, 100]
        # epochs = [10, 50, 100]
        # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        # init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
        # activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
        # neurons = [1, 5, 10, 15, 20, 25, 30]
        # weight_constraint = [1, 2, 3, 4, 5]
        # dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # param_grid = dict(batch_size=batch_size,optimizer=optimizer,epochs=epochs,init_mode=init_mode,activation=activation,neurons=neurons,weight_constraint=weight_constraint)
        # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,verbose=10)
        # grid_result = grid.fit(X,y)
        # # summarize results
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # means = grid_result.cv_results_['mean_test_score']
        # stds = grid_result.cv_results_['std_test_score']
        # params = grid_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))
    
        # fit model
        #model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
    
        #model.fit(X_train, y_train)
        #model.fit(X, y, epochs=180, batch_size=100,optimizer='Adamax',init_mode = 'lecun_uniform',activation = 'softmax',neurons = 100,weight_constraint = 3,dropout_rate = 0.9 ,verbose=2, shuffle=False)
        #model = create_model(neurons=100,optimizer='Adamax',init_mode='glorot_normal',activation='relu',weight_constraint=3)
        #model.fit(X, y, epochs=200,verbose=2,batch_size=1 shuffle=False)
        
        #model = create_model(neurons=neurons,optimizer=optimizer,init_mode=init_mode,activation=activation,weight_constraint=weight_constraint)
        #model.fit(X, y, epochs=epoch,verbose=2,batch_size=batch_size,shuffle=False)
        
        #raw_values = values1[2518:2543]
        #raw_values = values1[-24:]
        #test = values1[-24:]
        
        #Updated
        #test=values1[len(values1)-96:len(values1)-24]
        #test = values1[len(values2)-24:]
        #print(len(values1))
        #print(test)
        #

        #test = values2[len(values2)-48:len(values2)-24]
        #raw_values = values2[len(values2)-48:len(values2)-24]
        
        #Updated
        #raw_values = values2[len(values2)-168:len(values2)-96]
        #print(type(test))
        #print(test.shape)
        #
        
        #print(raw_values)
        # transform data to be stationary
        #diff_series = difference(raw_values, 1)
        #diff_values = diff_series.values
        #diff_values = diff_values.reshape(len(diff_values), 1)
        #print(diff_values)
        # rescale values to -1, 1
        
        
        #Updated
        #print("Raw values")
        #print(raw_values.shape)
        #scaler2 = MinMaxScaler(feature_range=(0, 1))
        #raw_values=raw_values.reshape(len(raw_values),1)
        #scaled_values = scaler2.fit_transform(diff_values)
        #scaled_values = scaler2.fit_transform(raw_values)
        #scaled_values = scaled_values.reshape(len(scaled_values), 1)
        #test1= scaled_values
        #print(test1)
        #

        #Updated
        #test1=test1.reshape(1,1,test1.shape[0])
        #print(test1.shape)
        #forecasts = list()
        #n_batch=200
        #forecast = model.predict(test1, batch_size=n_batch)
        #forecasts.append(forecast)
        #
        
        #n_batch=100
        # make forecasts
        #forecasts = make_forecasts(model, n_batch, test1, n_lag)
        #classifier.predict(X_test)
    
    
        #Updated
        #print(forecasts)
        #results=scaler2.inverse_transform(forecasts[0])
        #results = np.round(results)
        #results = results.tolist()
        #test = np.round(test)
        #test = test.tolist()
        #

        #Updated
        #results = [round(x) for x in results]
        #print(results)
        #evaluate_forecasts(test, results, n_lag, n_seq)
        #test = np.array(test)
        #test = test.tolist()
        #
        
        
        

        numpy.random.seed(7)

        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(df_orginal_total_requets)
                
        # split into train and test sets
        train_size = int(float(len(dataset)) * float(train_ratio))
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
        print(len(train), len(test))

        # reshape into X=t and Y=t+1
        look_back = 1
        trainX, trainY = create_dataset(train,  )
        testX, testY = create_dataset(test, look_back)

        # reshape input to be [samples, time steps, features]
        print(trainX.shape,testX.shape)
        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        
        if(os.path.isfile(key+'.json')):
            # load json and create model
            file_path = key+'.json'
            json_file = open(file_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            
            
            
            loaded_model.load_weights(key+".h5", by_name = True, skip_mismatch = True)
            print("Loaded model from disk")
            #model = create_model(neurons=100,optimizer='Adamax',init_mode='glorot_normal',activation='relu',weight_constraint=3)
            #model.fit(trainX, testX, epochs=200,verbose=2, shuffle=False)
            
            # evaluate loaded model on test data
            #loaded_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            loaded_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
            score = loaded_model.evaluate(testX, testY, verbose=10)
            print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
            
            #loaded_model.fit(trainX, trainY, epochs=int(float(epoch)),verbose=2,batch_size=int(float(batch_size)),shuffle=False)
    
            # make predictions
            trainPredict = loaded_model.predict(trainX)
            testPredict = loaded_model.predict(testX)
            print(trainPredict)
            print(testPredict)
            # invert predictions
            trainPredict = scaler.inverse_transform(trainPredict)
            trainY = scaler.inverse_transform([trainY])
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform([testY])
            # calculate root mean squared error
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
            print('Test Score: %.2f RMSE' % (testScore))
            
             # shift train predictions for plotting
            trainPredictPlot = numpy.empty_like(dataset)
            trainPredictPlot[:, :] = numpy.nan
            trainPredictPlot = trainPredict
             # shift test predictions for plotting
            testPredictPlot = numpy.empty_like(dataset)
            testPredictPlot[:, :] = numpy.nan
            testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
             # plot baseline and predictions
            plt.figure(figsize = (18,18))
            plt.plot(scaler.inverse_transform(dataset))
            plt.plot(trainPredictPlot)
            plt.plot(testPredictPlot)
            plt.show()

            results = np.round(testPredict[:,0])
            results = results.tolist()
        
            results_collection[key]=results

            test = np.array(testY[0])
            test = test.tolist()
            test_results_collection[key]=test

            # serialize model to JSON
            loaded_model_json = loaded_model.to_json()
            with open(key+".json", "w") as json_file:
                # json_file.write(loaded_model_json)
                json_file.write(simplejson.dumps(simplejson.loads(loaded_model_json), indent=4))
            # serialize weights to HDF5
            loaded_model.save_weights(key+".h5")
            print("Updated the model and saved to disk")
            
        else:
            model = create_model(neurons=neurons,optimizer=optimizer,init_mode=init_mode,activation=activation,weight_constraint=weight_constraint,look_back=look_back)
        
        
            # # define the checkpoint
            # filepath = key+".h5"
            # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')
            # callbacks_list = [checkpoint]    

            # # fit the model
            # #model.fit(x_train, y_train, epochs=5, batch_size=50, callbacks=callbacks_list)
            # model.fit(trainX, trainY, epochs=epoch,verbose=2,batch_size=batch_size,callbacks=callbacks_list,shuffle=False)
            # # load the model
            # new_model = load_model(key+".h5")
            # assert_allclose(model.predict(trainX),new_model.predict(trainX),1e-5)

            # # fit the model
            # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')
            # callbacks_list = [checkpoint]
            # #new_model.fit(x_train, y_train, epochs=5, batch_size=50, callbacks=callbacks_list)
        
            #model.fit(trainX, trainY, epochs=epoch,verbose=2,batch_size=batch_size,callbacks=callbacks_list,shuffle=False)
            #%%model.fit(trainX, trainY, epochs=int(float(epoch)),verbose=2,batch_size=int(float(batch_size)),shuffle=False)
            model.fit(trainX, trainY, epochs=int(float(epoch)), batch_size=1, verbose=2)


            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)
            # invert predictions
            trainPredict = scaler.inverse_transform(trainPredict)
            trainY = scaler.inverse_transform([trainY])
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform([testY])
            # calculate root mean squared error
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
            print('Test Score: %.2f RMSE' % (testScore)) 

            # serialize model to JSON
            model_json = model.to_json()
            with open(key+".json", "w") as json_file:
                # json_file.write(model_json)
                json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))
            # serialize weights to HDF5
            model.save_weights(key+".h5")
            print("Saved model to disk")    

            results = np.round(testPredict[:,0])
            results = results.tolist()
        
            results_collection[key]=results

            test = np.array(testY[0])
            test = test.tolist()
            test_results_collection[key]=test

        
            # # shift train predictions for plotting
            # trainPredictPlot = numpy.empty_like(dataset)
            # trainPredictPlot[:, :] = numpy.nan
            # trainPredictPlot = trainPredict
            # # shift test predictions for plotting
            # testPredictPlot = numpy.empty_like(dataset)
            # testPredictPlot[:, :] = numpy.nan
            # testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
            # # plot baseline and predictions
            # plt.figure(figsize = (18,18))
            # plt.plot(scaler.inverse_transform(dataset))
            # plt.plot(trainPredictPlot)
            # plt.plot(testPredictPlot)
            # plt.show()    
        
    return render(request,'api/results.html',{'results':json.dumps(results),'region_variable':region_variable,'actual_data':json.dumps(test)})


    
