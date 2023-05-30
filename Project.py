################(a) Data Loading:##########
import pandas as pd
data_set = pd.read_csv('https://query.data.world/s/ixteovs4vu46d7ulr377xf2ajf52hi?dws=00000', sep=",",header=0)  
# print(data_set)
# data_set=data_set.head(204)



# ############### (b) Data Exploration: #######   
# data_set = data_set.dropna()
# print(data_set.isnull().sum())
# data_set=data_set.drop_duplicates()
# print(data_set.duplicated().sum())
# #detect wrong data in the specific column in dataset
# import matplotlib.pyplot as plt
# x=[i for i in range(data_set.shape[0])]
# plt.plot(x,data_set['carheight'])
# plt.show()

# ###ابراهيم حسن خلف


############### (c) Data Visualization: ####### 
#split data catigrocal and continues to plot the continues data
data_con=data_set.select_dtypes(exclude='object')
data_cat=data_set.select_dtypes(["object"])

#Encode categrocal data 
from sklearn.preprocessing import LabelEncoder
for i in range (data_cat.shape[1]):
    pre_data=LabelEncoder()
    data_cat[data_cat.columns[i]]=pre_data.fit_transform(data_cat[data_cat.columns[i]])
#Scale continues data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_data=scaler.fit_transform(data_con)
print(scaled_data)

# #أحمد عادل سيد

# ##split data catigrocal and continues to X,y
X_con=scaled_data[:,:-1]
y_con=scaled_data[:,-1]
X_cat=data_cat.iloc[:,1:-1].values
y_cat=data_cat.iloc[:,3].values

#plot the corrolation(heatmap)
data_con_r = data_con.corr()
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(data_con_r,annot=True)  
plt.show()



############### (d) Descriptive Analysis: ########
#plot the pair plot
sns.pairplot(data_con_r)
plt.show()


############### (e) Interpretation and Findings:#######

# *The data we use is a high accurate data. it don't have any empty cell or wrong data ; only duplicated row(i remove this row)
# *also there are a relation(corrolation) between all columns such as positive corrolation between "enginesize" and "horsepower" 

#أحمد محمد عبد القادر

############### splitting data and Accuracy#######
# import numpy as np
# from sklearn.model_selection import train_test_split
# X_train_con,X_test_con,y_train_con,y_test_con=train_test_split(X_con,y_con,test_size=0.3)
# from sklearn.linear_model import LinearRegression #first we import the class "LinearRegression"
# model=LinearRegression() #store the class in a variale to be a methode
# model.fit(X_train_con,y_train_con) #train the model by the function "fit" with 70% of data
# Accuracy=model.score(X_test_con,y_test_con) #get the accuracy by test the data by compare the train data with the test data
# print("Accuracy = ",Accuracy)



# ############## Evaluation #######
# #get Mean Absolute Error
# from sklearn.metrics import mean_absolute_error
# y_pred_con=model.predict(X_test_con)
# MAE = mean_absolute_error(y_test_con,y_pred_con)
# print("Mean Absolute Error= ",MAE)

# #calculate accuracy of continues data
# model_con=LinearRegression()
# model_con.fit(X_train_con,y_train_con)
# Acc=model_con.score(X_test_con,y_test_con)
# print("Accuracy =",Acc)



###### A brief report ######




# Topic: Car prices dataset

# Dataset Description: The dataset contains information about the prices of different cars, as well as their features such as horsepower, fuel type, and body style. The dataset was loaded from a CSV file using the pandas library.

# Approach to Analysis:
# (a) Data Loading: The first step was to load the car prices dataset using pandas. The dataset was stored in a CSV file
# (b) Data Exploration: The next step involved exploring the dataset to check for missing values and duplicates. The missing values were dropped from the dataset using the dropna() function, and the duplicates were removed using the drop_duplicates() function. The code also included a plot to detect any wrong data in the specific column in the dataset.
# (c) Data Visualization: The third step. The code split the data into categorical and continuous variables, and then plotted the correlation to visualize the correlations between the continuous variables.
# (d) Descriptive Analysis: The final step involved performing a descriptive analysis of the dataset using a pair plot to visualize the relationships between the variables.

#I get the dataset from this website https://data.world/ 


#أحمد ناصر الدين