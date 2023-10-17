# import numpy as np
# x=np.array([[1,2],[3,4]])
# y=np.array([[5,6],[7,8]])


# v=np.array([9,10])
# w=np.array([11,12])

# print(np.dot(v,w),"\n")

# print(np.dot(x,v),"\n")

# print(np.dot(x,y))


# from scipy.misc import 
# import imageio.v2 as imageio
# from imageio import imread,imsave
# file="c:\\Users\\DELL\\OneDrive\\Pictures\\Agri.jpg"

# format="jpg"
# img=imread(file)

# print(img.dtype,img.shape)

# img_tint=img*[1,0.45,0.3]

# imsave(file)

# from sklearn import datasets
# from sklearn import metrics

# from sklearn.tree import DecisionTreeClassifier

# dataset=datasets.load_iris()

# model=DecisionTreeClassifier()

# model.fit(dataset.data,dataset.target)

# print(model)

# expected=dataset.target

# predicted=model.predict(dataset.data)


# print(metrics.classification_report(expected,predicted))
# print(metrics.confusion_matrix(expected,predicted))



# import theano
# import theano.tensor as T
# x=T.dmatrix('x')

# s=1/(1+T.exp(-x))

# logistic =theano.function([x],s)
# logistic([[0,1],[1,2]])


# A = [[12, 7, 3],
#     [4, 5, 6],
#     [7, 8, 9]]
 
# # take a 3x4 matrix
# B = [[5, 8, 1, 2],
#     [6, 7, 3, 0],
#     [4, 5, 9, 1]]

# result = [[sum(a * b for a, b in zip(A_row, B_col)) 
#                         for B_col in zip(*B)]
#                                 for A_row in A]
 

# for r in result:
    # print(r)


# n=int(input("entr ht enu"))
# b=1
# if(n%2==0):

#     n=n%2
#     c=n+1
#     print(n)
# else:
#     n=n%2
#     c=n
#     print(f"\tn")

from sklearn.datasets import load_iris
import pandas as pd


# dataset=pd.DataFrame(load)
# data=load_iris(as_frame=True,return_X_y=True)


# data=pd.DataFrame(data)
# # print(data.head)

# # print(data['Species'].unique())

# from sklearn import preprocessing

# label_encoder=preprocessing.LabelEncoder()

# data['Species']=label_encoder.fit_transform(data['Species'])
# print(data['Species'].unique())
# # print(pd.DataFrame(data,dtype=object))



import pandas as pd
import numpy as np

data= pd.read_csv(r'c:\Users\DELL\OneDrive\Desktop\dataset\Rainfall_data.csv')

data=pd.DataFrame(data)
print("\n")
print(data.info)
print(data.isnull().sum())
print(data.describe())
data=data.drop('Day',axis=1)
data=data.dropna()
print(data.shape)

from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
# sns.boxplot(data['Precipitation'])

# ax=plt.subplot(6,4,1)

# ax.scatter(data['Precipitation'],data['Relative Humidity'])
# ax.set_xlabel('(Relative humidiy)')
# plt.show()

# print(np.where(data['Precipitation']>800))

print("relative\n")
q1,q3=np.percentile(data["Relative Humidity"],[25,75])
iqr=q3-q1
lower_bound=q1-(1.5*iqr)
upper_bound=q3+(1.5*iqr)

clean_data=data[(data["Relative Humidity"]>=lower_bound)&(data["Relative Humidity"]<=upper_bound)]
print("///////////////////////////////////////////////////////////")
q1,q3=np.percentile(data["Specific Humidity"],[25,75])
iqr=q3-q1
lower_bound=q1-(1.5*iqr)
upper_bound=q3+(1.5*iqr)



clean_data=data[(data["Specific Humidity"]>=lower_bound)&(data["Specific Humidity"]<=upper_bound)]
# print("\n")
# print(clean_data.info)
# print(clean_data.shape)

print("///////////////////////////////////////////////////////////")

q1,q3=np.percentile(data["Temperature"],[25,75])
iqr=q3-q1
lower_bound=q1-(1.5*iqr)
upper_bound=q3+(1.5*iqr)
print("\n")
# print(clean_data.info)
# print(clean_data.shape)

clean_data=data[(data["Temperature"]>=lower_bound)&(data["Temperature"]<=upper_bound)]
print("\n")

# print(clean_data.info)
# print(clean_data.shape)

corr=data.corr()
plt.figure(dpi=130)
sns.heatmap(data.corr(),annot=True,fmt='.2f')
# plt.show()

corr['Precipitation'].sort_values(ascending=False)
# print(corr)

# plt.pie(data.Precipitation.value_counts(),
        
#         autopct='%.f',shadow=True)

# plt.title('Outcome Proportionality')
# plt.show()

x=pd.DataFrame(data.drop(columns=['Precipitation']))

y=pd.DataFrame(data['Precipitation'])



scaler =preprocessing.MinMaxScaler(feature_range=(0,1))
rescale=scaler.fit_transform(x)
# print(rescale[:5])

# from sklearn.feature_extraction import DictVectorizer

# vec=DictVectorizer()

# vec.fit_transform(data)
# print(vec.get_feature_names_out())

from sklearn.linear_model import LinearRegression

model=LinearRegression()

model.fit(x,y)


prediction=model.predict(x)

print(prediction)






