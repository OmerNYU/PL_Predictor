import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv('premier-league-matches.csv')

print(df.shape)    #how many rows and columns are there
print(df.head())   #first 5 rows of the dataframe
print(df.dtypes)   #data types of the columns
print(df.isnull().sum())  #check for null values


# decode the result column first
df['result'] = df['FTR'].map({'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'})
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# build pre-match rolling averages (what we'd actually know before kickoff)
df['home_goals_avg'] = df.groupby('Home')['HomeGoals'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df['away_goals_avg'] = df.groupby('Away')['AwayGoals'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df['home_conceded_avg'] = df.groupby('Home')['AwayGoals'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df['away_conceded_avg'] = df.groupby('Away')['HomeGoals'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df = df.dropna()

le_home = LabelEncoder()
le_away = LabelEncoder()
le_result = LabelEncoder()

df['home_encoded'] = le_home.fit_transform(df['Home'])
df['away_encoded'] = le_away.fit_transform(df['Away'])
df['result_encoded'] = le_result.fit_transform(df['result'])

print(df['result'].value_counts())
print(df['result'].value_counts(normalize=True).round(3))

# quick correlation check
print(df.corr(numeric_only=True)['result_encoded'].sort_values())


#visualize the correlation

df['HomeGoals'].hist(bins=10)
plt.title('Home Goals Distribution')
plt.show()

features = [
    'home_encoded', 'away_encoded',
    'home_goals_avg', 'away_goals_avg',
    'home_conceded_avg', 'away_conceded_avg'
]

X = df[features]
y = df['result_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)