import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('premier-league-matches.csv')

print(df.shape)    #how many rows and columns are there
print(df.head())   #first 5 rows of the dataframe
print(df.dtypes)   #data types of the columns
print(df.isnull().sum())  #check for null values


#decode FTR into readable labels

df['result'] = df['FTR'].map({'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'})

print(df['result'].value_counts())
print(df['result'].value_counts(normalize=True).round(3))


#for numeric features

df.describe()

#encode results so we can run the correlation

le_temp = LabelEncoder()
df['result_encoded'] = le_temp.fit_transform(df['result'])

#quick correlation check
df.corr(numeric_only=True)['result_encoded'].sort_values()


#visualize the correlation

df['HomeGoals'].hist(bins=10)
plt.title('Home Goals Distribution')
plt.show()