import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("diabetes.csv");
# print(data);

#Head of datasets
#Display the some top values 
# print(data.head())

#Describe function means detail of datasets
# print(data.describe())
#Check the null value from the datasets 
# print(data.isna().sum()); #sum means total null value in datasets

#To find the duplicate data 
#  # sum means total duplicate value

null_rows = data[data.isnull().any(axis=1)]

# Display columns with null values
# print("empty row",null_rows)

null_columns = data.columns[data.isnull().any()]

# Display columns with null values
# print("empty column",null_columns)

# x =data.iloc[: , :].values #all data will be fetch and store in variable x 
#Empty rows and column are ['SkinThickness', 'Insulin', 'Age'] ; after the operation

# Specify columns with missing values
columns_to_impute = ['SkinThickness', 'Insulin', 'Age']

# Apply SimpleImputer to replace missing values with median
imputer = SimpleImputer(strategy='median')  # You can use 'mean' instead of 'median' if preferred or most frequent value 
# But i prefer median for it safe from float value 

# Fit and transform the selected columns
data[columns_to_impute] = imputer.fit_transform(data[columns_to_impute])

# Now 'data' should have missing values replaced with median  in specified columns
# print("Data after imputation:")
# print(data.head())

data.to_csv("diabetes_Full_clean_data.csv", index=False)
print("Imputed data saved to 'diabetes_imputed.csv'")

pre_data =  pd.read_csv("diabetes_Full_clean_data.csv");

# print(pre_data);

#Visualize the data on graphs

# plt.figure(figsize = (12,6)) # total width is 12 fit and height 6 fit 
# sns.countplot(x='Outcome' ,data=pre_data);
# plt.show()

# find the outliers and it was unexpected values 

# Get column names
column_names = data.columns
# print(column_names)

plt.figure(figsize=(12,12))

# for i ,col in enumerate(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']):
#     plt.subplot(3,3 ,i+1)
#     #space on a main graph so easily understand it 
#     sns.boxenplot(x=col , data = pre_data);
#     plt.title(col)
# plt.tight_layout()
# plt.show()

#Pair plot 

# sns.pairplot(pre_data,hue="Outcome" )
# plt.show();

# generate the histograph

# for i ,col in enumerate(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']):
#     plt.subplot(3,3 ,i+1)
#     #space on a main graph so easily understand it 
#     sns.histplot(x=col,data=pre_data, kde = True)

# plt.show()

#heat map 

plt.figure(figsize=(12, 6))
sns.heatmap(pre_data.corr(), vmin=-1.0, center=0, cmap='RdBu_r', annot=True)
plt.title("Correlation Heatmap")
plt.show()

print(pre_data.corr())