#!/usr/bin/env python
# coding: utf-8

# In[8]:


#question 1
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

file_path = "Baseball Stats.xlsx" 
data = pd.read_excel(file_path, sheet_name=None)

batting_stats = data["Ohtani 2024 Reg S Batting Stats"]
swing_stats = data["Ohtani 2024 Reg S Swing Stats"] 

#columns for comparison
batting_cols = ['H', 'R', 'RBI']
swing_cols = ['EV (MPH)', 'LA (°)', 'Dist (ft)', 'Pitch (MPH)']  

#filter the columns
batting_data = batting_stats[batting_cols] 
swing_data = swing_stats[swing_cols]  

#fix data with missing values
batting_data = batting_data.dropna()
swing_data = swing_data.dropna()

#combine swing and batting data
aligned_data = pd.concat([batting_data.reset_index(drop=True), swing_data.reset_index(drop=True)], axis=1)

X = aligned_data[swing_cols]  #swing data is input
targets = {
    'Hits': aligned_data['H'],  #Hits
    'RBIs': aligned_data['RBI'],  #Runs Batted In
    'Runs': aligned_data['R']  #Runs 
}

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = {}

#to train and evaluate the models
for target_name, y in targets.items():
    #drop rows with missing values in the current target variable
    y = y.dropna()
    X_cleaned = X.loc[y.index] 
    X_scaled_cleaned = scaler.fit_transform(X_cleaned)

    #split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_cleaned, y, test_size=0.2, random_state=1)

    #apply ElasticNet regression with cross validation
    elastic_net = ElasticNetCV(
        cv=5, 
        random_state=1,
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0]  #range of l1 ratios
    )
    elastic_net.fit(X_train, y_train)  #fit the model to the training data

    #predict target values for the test set
    y_pred = elastic_net.predict(X_test)

    #evaluate the model using Mean Squared Error and R-squared metrics
    mse = mean_squared_error(y_test, y_pred)  
    r2 = r2_score(y_test, y_pred) 

    #get the coefficients
    coefficients = pd.DataFrame({
        'Swing Metric': swing_cols, 
        'Coefficient': elastic_net.coef_
    })

    results[target_name] = {
        'MSE': mse,
        'R^2': r2,
        'Coefficients': coefficients.sort_values(by='Coefficient', ascending=False) 
    }

for target, result in results.items():
    print(f"Results for {target}:")
    print(result['Coefficients'])  
    print(f"Mean Squared Error (MSE): {result['MSE']}")  
    print(f"R-squared (R^2): {result['R^2']}\n") 


# In[6]:


#question 2
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

file_path = "Baseball Stats.xlsx"  
data = pd.read_excel(file_path, sheet_name=None) 

players = {
    "Ohtani": {
        "Swing Stats": data["Ohtani 2024 Reg S Swing Stats"], 
        "Batting Stats": data["Ohtani 2024 Reg S Batting Stats"] 
    },
    "Judge": {
        "Swing Stats": data["Judge 2024 Reg S Swing Stats"], 
        "Batting Stats": data["Judge 2024 Reg S Batting Stats"]  
    },
    "Stanton": {
        "Swing Stats": data["Stanton 2024 Reg S Swing Stats"],  
        "Batting Stats": data["Stanton 2024 Reg S Batting Stat"] 
    },
    "Cruz": {
        "Swing Stats": data["Cruz 2024 Reg S Swing Stats"], 
        "Batting Stats": data["Cruz 2024 Reg S Batting Stats"] 
    }
}

#columns for comparison
swing_cols = ['EV (MPH)', 'LA (°)', 'Dist (ft)', 'Pitch (MPH)']  #swing data
batting_cols = ['H', 'R', 'RBI']  #batting data

all_combined_stats = []
player_labels = [] 

#loop through each player's data
for player, stats in players.items():
    swing_stats = stats["Swing Stats"][swing_cols].dropna()  #clean swing stats
    batting_stats = stats["Batting Stats"][batting_cols].dropna()  #clean batting stats
    
    #make data same size
    min_len = min(len(swing_stats), len(batting_stats))
    swing_stats = swing_stats.iloc[:min_len]  
    batting_stats = batting_stats.iloc[:min_len]  
    
    #combine stats to single set
    combined_stats = pd.concat([swing_stats.reset_index(drop=True), 
    batting_stats.reset_index(drop=True)], axis=1)
    all_combined_stats.append(combined_stats)  
    player_labels.extend([player] * len(combined_stats))

#combine all data
combined_data = pd.concat(all_combined_stats, axis=0).reset_index(drop=True)

#handle missing or infinite values in the data
combined_data = combined_data.replace([float('inf'), -float('inf')], np.nan).dropna()

scaler = StandardScaler()
combined_data_scaled = scaler.fit_transform(combined_data)

#apply KMeans clustering to the standardized data
kmeans = KMeans(n_clusters=4, random_state=1)  
kmeans.fit(combined_data_scaled)  

combined_data['Cluster'] = kmeans.labels_ 
combined_data['Player'] = player_labels  

#summarize clusters by counting points per player in each cluster
cluster_summary = combined_data.groupby(['Cluster', 'Player']).size().unstack(fill_value=0)

#visualize clusters 
plt.figure(figsize=(10, 6)) 
for cluster in range(4):  
    cluster_points = combined_data[combined_data['Cluster'] == cluster]  
    plt.scatter(cluster_points['EV (MPH)'], cluster_points['H'], label=f'Cluster {cluster}')  

#labels
plt.xlabel('Exit Velocity (MPH)')
plt.ylabel('Hits')
plt.title('Swing and Batting Stats Clusters for Ohtani, Judge, Stanton, and Cruz')
plt.legend()
plt.show()  

print("Cluster Summary:")
print(cluster_summary)  

centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=combined_data.columns[:-2])
print("Cluster Centroids:")
print(centroids)


# In[29]:


#question 3
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

file_path = "Baseball Stats.xlsx" 
data = pd.read_excel(file_path, sheet_name=None)


dodgers_reg_stats = data["Dodgers 2024 Reg S Stats"]
yankees_reg_stats = data["Yankees 2024 Reg S Stats"]


dodgers_reg_stats['Team'] = 'Dodgers'
yankees_reg_stats['Team'] = 'Yankees'

#combine the data into one dataset
combined_stats = pd.concat([dodgers_reg_stats, yankees_reg_stats], axis=0).reset_index(drop=True)

#specific stats used 
swinging_stats_cols = ['AB', 'H', '2B', '3B', 'HR', 'SO', 'BB']

#handle missing values by dropping invalid rows
X = combined_stats[swinging_stats_cols].dropna()
y = combined_stats['Team'].loc[X.index]

#standardize the stats
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#labels for SVC
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#train Linear SVC
svc = SVC(kernel='linear', random_state=1)
svc.fit(X_pca, y_encoded)

#define ranges
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

#predict and reshape points
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#configure plot
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm', levels=np.arange(-0.5, 2, 1))

#plot
team_labels = label_encoder.classes_
colors = ['blue', 'orange']

for i, (team, color) in enumerate(zip(team_labels, colors)):
    team_data = X_pca[y_encoded == i]
    plt.scatter(
        team_data[:, 0],
        team_data[:, 1],
        label=team,
        color=color,
        edgecolors='k',
        alpha=0.7
    )

#add labels
plt.title('Linear SVC Decision Boundary: Dodgers vs Yankees')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()

explained_variance = pca.explained_variance_ratio_
print(f"Variance by PCA1: {explained_variance[0]:.2f}")
print(f"Variance by PCA2: {explained_variance[1]:.2f}")

#pca loadings
loadings = pd.DataFrame(
    pca.components_,  
    columns=swinging_stats_cols,  
    index=['PCA1', 'PCA2']  
)

print(loadings)


# In[30]:


#question 4
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

file_path = "Baseball Stats.xlsx" 
data = pd.read_excel(file_path, sheet_name=None)

dodgers_ws_stats = data["Dodgers 2024 WS Stats"]
dodgers_reg_stats = data["Dodgers 2024 Reg S Stats"]
yankees_ws_stats = data["Yankees 2024 WS Stats"]
yankees_reg_stats = data["Yankees 2024 Reg S Stats"]

dodgers_ws_stats['Game Type'] = 'World Series'
dodgers_reg_stats['Game Type'] = 'Regular Season'
yankees_ws_stats['Game Type'] = 'World Series'
yankees_reg_stats['Game Type'] = 'Regular Season'

#combine the data into one dataset
dodgers_stats = pd.concat([dodgers_ws_stats, dodgers_reg_stats], axis=0)
yankees_stats = pd.concat([yankees_ws_stats, yankees_reg_stats], axis=0)
combined_stats = pd.concat([dodgers_stats, yankees_stats], axis=0).reset_index(drop=True)

#get specific data types
swinging_stats_cols = ['AB', 'H', '2B', '3B', 'HR', 'SO', 'BB']
X = combined_stats[swinging_stats_cols]
#specifiy by game type
y = combined_stats['Game Type']  

#handle missing values by dropping invalid rows
X = X.dropna()
y = y.loc[X.index]

#standardize the swinging stats
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#apply Randomized PCA
pca = PCA(n_components=2, random_state=1)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
pca_df['Game Type'] = y.values

#visualize results
plt.figure(figsize=(10, 6))
for game_type, color in zip(['World Series', 'Regular Season'], ['red', 'blue']):
    subset = pca_df[pca_df['Game Type'] == game_type]
    plt.scatter(subset['PCA1'], subset['PCA2'], label=game_type, alpha=0.7, edgecolors='k')

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA of Swinging Stats: World Series vs Regular Season')
plt.legend()
plt.show()


explained_variance = pca.explained_variance_ratio_
print(f"Variance by PCA1: {explained_variance[0]:.2f}")
print(f"Variance by PCA2: {explained_variance[1]:.2f}")

#pca loadings
loadings = pd.DataFrame(
    pca.components_,  
    columns=swinging_stats_cols,  
    index=['PCA1', 'PCA2']  
)

print(loadings)


# In[5]:


#question 5
import pandas as pd
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

file_path = "Baseball Stats.xlsx"
data = pd.read_excel(file_path, sheet_name=None)

pitch_stats = data["Pitch Stats"]

#create isomap
def create_isomap(data, features, pitch_type_column, title, axis_labels):
    #filter data
    filtered_data = data[features + [pitch_type_column]].dropna()
    
    #extract features and labels
    X = filtered_data[features]
    y = filtered_data[pitch_type_column]
    
    #standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    isomap = Isomap(n_neighbors=5, n_components=2)
    X_isomap = isomap.fit_transform(X_scaled)
    
    isomap_df = pd.DataFrame(data=X_isomap, columns=['Isomap1', 'Isomap2'])
    isomap_df['Pitch Type'] = y.values
    
    #plot
    plt.figure(figsize=(12, 8))
    for pitch in isomap_df['Pitch Type'].unique():
        subset = isomap_df[isomap_df['Pitch Type'] == pitch]
        plt.scatter(subset['Isomap1'], subset['Isomap2'], label=pitch, alpha=0.7, edgecolors='k')
    
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

#SLG and Whiff %
create_isomap(
    pitch_stats,
    features=['SLG', 'Whiff %'],
    pitch_type_column='Pitch',
    title='SLG vs Whiff % by Pitch Type',
    axis_labels=['SLG', 'Whiff%']
)

#K% and Put Away %
create_isomap(
    pitch_stats,
    features=['K%', 'Put Away %'],
    pitch_type_column='Pitch',
    title='K% vs Put Away % by Pitch Type',
    axis_labels=['K%', 'Put Away%']
)

