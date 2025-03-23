import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def preprocess(self):
        """Preprocess the data."""
        # Convert the 'Date' column to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d/%m/%Y')
        
        # Calculate the goal difference and the odds ratio
        self.df['GoalDiff'] = self.df['FTHG'] - self.df['FTAG']
        self.df['OddsRatio'] = self.df['B365H'] / self.df['B365A']  # Example odds ratio
        
        # Drop rows with missing values in the relevant columns (GoalDiff and OddsRatio)
        self.df.dropna(subset=['GoalDiff', 'OddsRatio'], inplace=True)

        # Encode categorical features (for example, teams)
        self.df['HomeTeam'] = self.df['HomeTeam'].astype('category').cat.codes
        self.df['AwayTeam'] = self.df['AwayTeam'].astype('category').cat.codes
        
        # Fill missing values for numeric columns
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        
        return self.df

class FootballPredictor:
    def __init__(self, df):
        self.df = df
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def load_and_preprocess_data(self):
        """Load and preprocess data."""
        preprocessor = DataPreprocessor(self.df)
        self.df = preprocessor.preprocess()
        
    def train_model(self):
        """Train the Random Forest model."""
        X = self.df[['GoalDiff', 'OddsRatio', 'HomeTeam', 'AwayTeam']]  # Features
        y = self.df['FTR']  # Target variable (Home Win, Draw, Away Win)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Model evaluation
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
    
    def predict_match(self, new_match):
        """Predict match result for new data."""
        # Assuming new_match is a DataFrame with similar features
        new_match_scaled = new_match[['GoalDiff', 'OddsRatio', 'HomeTeam', 'AwayTeam']]
        prediction = self.model.predict(new_match_scaled)
        return prediction


# Usage
url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
df = pd.read_csv(url)

football_predictor = FootballPredictor(df)
football_predictor.load_and_preprocess_data()
football_predictor.train_model()

# Predicting for a new match
new_match = pd.DataFrame({
    'GoalDiff': [1],
    'OddsRatio': [2.5],
    'HomeTeam': [1],  # Example encoded team value
    'AwayTeam': [3]   # Example encoded team value
})
predicted_result = football_predictor.predict_match(new_match)
print(f"Predicted result: {predicted_result}")
