import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv('top_100_spotify_songs_2025.csv')

# Select features and target
X = df[['Spotify_Streams_Millions', 'Duration_Seconds']]
y = df['Popularity_Score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test data
pred = model.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, pred)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R² Score:", r2)
