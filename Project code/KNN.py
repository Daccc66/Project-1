import pandas as pd
from sklearn.model_selection import train_test_split

dataset=pd.read_csv('new_data.csv')
dataset.isna().sum() #no missing data
dataset.columns
dataset.columns=['gameId','creationTime','gameDuration','seasonId','winner','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
x=dataset.iloc[:,~dataset.columns.isin(['winner','gameId','creationTime','seasonId'])].values
y=dataset.loc[:,'winner'].values
winner_is_team1 = dataset.loc[y == 1]
winner_is_team2 = dataset.loc[y == 2]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=90)

dt2 = pd.read_csv('test_set.csv')
dt2.isna().sum() #no missing data
dt2.columns
dt2.columns=['gameId','creationTime','gameDuration','seasonId','winner','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
xx=dt2.iloc[:,~dataset.columns.isin(['winner','gameId','creationTime','seasonId'])].values
yy=dt2.loc[:,'winner'].values
winner_is_team1 = dataset.loc[y == 1]
winner_is_team2 = dataset.loc[y == 2]

from sklearn.neighbors import KNeighborsClassifier
KNN= KNeighborsClassifier(n_neighbors = 3)
KNN.fit(x_train, y_train)
y_predict_KNN = KNN.predict(xx)
from sklearn.metrics import accuracy_score
print(accuracy_score(yy, y_predict_KNN))
import time
start = time.time()
for _ in range(100000000):
    pass
end = time.time()
print((end-start))