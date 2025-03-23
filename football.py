import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class FootballDataLoader:
    def __init__(self, url):
        self.url = url
        self.df = None
    
    def load_data(self):
        """Load the data from the given URL."""
        self.df = pd.read_csv(self.url)
        return self.df

class DataPreprocessor:
    def __init__(self, df):
        self.df = df
    
    def preprocess(self):
        """Handle missing values and perform feature engineering."""
        
        # Identify numeric columns
        numeric_columns = self.df.select_dtypes(include='number').columns
        
        # Handle missing values by filling with median for numeric columns
        self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].median())
        
        # Feature Engineering: Goal Difference and Odds Ratio
        self.df['GoalDiff'] = self.df['FTHG'] - self.df['FTAG']
        self.df['OddsRatio'] = self.df['B365H'] / self.df['B365A']
        
        # Encoding categorical columns (example: HomeTeam and AwayTeam)
        self.df['HomeTeam'] = self.df['HomeTeam'].astype('category').cat.codes
        self.df['AwayTeam'] = self.df['AwayTeam'].astype('category').cat.codes
        
        # Map the target variable (FTR) to numeric values (H = Home Win, D = Draw, A = Away Win)
        self.df['FTR'] = self.df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
        
        return self.df

class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = LogisticRegression(max_iter=1000)
    
    def train(self):
        """Train the model."""
        self.model.fit(self.X, self.y)
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model."""
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

class FootballPredictor:
    def __init__(self, url):
        self.url = url
        self.df = None
        self.X = None
        self.y = None
        self.model_trainer = None
    
    def load_and_preprocess_data(self):
        # Load the dataset
        loader = FootballDataLoader(self.url)
        self.df = loader.load_data()
        
        # Preprocess the data
        preprocessor = DataPreprocessor(self.df)
        self.df = preprocessor.preprocess()
        
        # Feature selection
        self.X = self.df[['HomeTeam', 'AwayTeam', 'GoalDiff', 'OddsRatio']]
        self.y = self.df['FTR']
    
    def train_model(self):
        trainer = ModelTrainer(self.X, self.y)
        trainer.train()
        self.model_trainer = trainer
    
    def evaluate_model(self, X_test, y_test):
        accuracy, report = self.model_trainer.evaluate(X_test, y_test)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
    
    def make_prediction(self, new_match):
        predicted_result = self.model_trainer.predict(new_match)
        return predicted_result[0]

# ---------------------------------------
# Main Execution
# ---------------------------------------

# Define the URL for the dataset
url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"

# Initialize the Football Predictor
football_predictor = FootballPredictor(url)

# Load and preprocess data
football_predictor.load_and_preprocess_data()

# Split the data into train and test sets
X = football_predictor.X
y = football_predictor.y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
football_predictor.train_model()

# Evaluate the model
football_predictor.evaluate_model(X_test, y_test)

# Example prediction for a new match
new_match = pd.DataFrame({'HomeTeam': [10], 'AwayTeam': [5], 'GoalDiff': [2], 'OddsRatio': [1.1]})
predicted_result = football_predictor.make_prediction(new_match)
print(f"Predicted Result (0=Home Win, 1=Draw, 2=Away Win): {predicted_result}")
