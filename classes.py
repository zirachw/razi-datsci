import numpy as np
import pandas as pd

# Sklearn is too heavy, vercel can't handle it
class ManualMinMaxScaler():
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.min_range_ = feature_range[0]
        self.max_range_ = feature_range[1]

    def fit(self, X):
        # Compute the minimum and maximum values for each feature
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        
        # Compute the scaling factor
        self.scale_ = (self.max_range_ - self.min_range_) / (self.max_ - self.min_)
        return self

    def transform(self, X):
        # Apply the scaling to each feature
        X_scaled = (X - self.min_) * self.scale_ + self.min_range_
        return X_scaled

    def fit_transform(self, X):
        # Fit to data, then transform it
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        # Reverse the scaling
        X_original = (X_scaled - self.min_range_) / self.scale_ + self.min_
        return X_original

class DecisionStump():
    def __init__(self, max_depth=None, random_state=None):
        self.max_depth = max_depth
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
    
    def fit(self, X, y, weights):
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y
        self.feature_idx = None
        self.threshold = None
        self.polarity = None
        self.alpha = None
        min_error = float('inf')
        
        m, n = X.shape
        
        for feature in range(n):
            feature_values = np.unique(X[:, feature])
            
            for threshold in feature_values:
                for polarity in [1, -1]:
                    predictions = self.predict(X, feature, threshold, polarity)
                    error = np.sum(weights * (predictions != y))
                    
                    if error < min_error:
                        min_error = error
                        self.feature_idx = feature
                        self.threshold = threshold
                        self.polarity = polarity
                        self.alpha = 0.5 * np.log((1.0 - error) / (error + 1e-10))
    
    def predict(self, X, feature=None, threshold=None, polarity=None):
        X = X.values if isinstance(X, pd.DataFrame) else X
        if feature is None or threshold is None or polarity is None:
            feature = self.feature_idx
            threshold = self.threshold
            polarity = self.polarity
        
        predictions = np.ones(X.shape[0])
        if polarity == 1:
            predictions[X[:, feature] < threshold] = -1
        else:
            predictions[X[:, feature] >= threshold] = -1
        
        return predictions

class ManualAdaBoostClassifier():
    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.alphas = []
        self.stumps = []
        self.rng = np.random.default_rng(random_state)
    
    def fit(self, X, y):
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y
        m, n = X.shape
        weights = np.ones(m) / m
        y = np.where(y == 1, 1, -1)  # Convert to {-1, 1}
        
        for _ in range(self.n_estimators):
            stump = DecisionStump(random_state=self.random_state)
            stump.fit(X, y, weights)
            predictions = stump.predict(X)
            error = np.sum(weights * (predictions != y))
            
            if error == 0:
                break
            
            alpha = self.learning_rate * stump.alpha
            weights *= np.exp(alpha * (predictions != y))
            weights /= np.sum(weights)
            
            self.stumps.append(stump)
            self.alphas.append(alpha)
    
    def predict(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else X
        m = X.shape[0]
        predictions = np.zeros(m)
        
        for stump, alpha in zip(self.stumps, self.alphas):
            predictions += alpha * stump.predict(X)
        
        return np.where(predictions > 0, 1, 0)
    
    def predict_proba(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else X
        m = X.shape[0]
        predictions = np.zeros(m)
        
        for stump, alpha in zip(self.stumps, self.alphas):
            predictions += alpha * stump.predict(X)
        
        # Convert to probabilities
        proba = 1 / (1 + np.exp(-predictions))
        return np.vstack([1 - proba, proba]).T
    
# Custom transformers from pipeline
class DateTransformer():
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X['Dt_Customer'] = pd.to_datetime(X['Dt_Customer']).dt.to_period('M')
        return X
    
class RemoveSkewness():
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        mnt_features = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        purchases_features = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
        skewed = mnt_features + purchases_features

        for feature in skewed:
            if feature == 'NumCatalogPurchases' or feature == 'NumWebPurchases' :
                X[feature] = np.sqrt(X[feature])
            else :
                X[feature] = np.log1p(X[feature])
        return X
    
class FeatureEngineering():
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        # Enrollment year
        X['EnrollmentYear']  = X['Dt_Customer'].dt.year
        X['EnrollmentYears'] = 2024 - X['EnrollmentYear']

        # Joint marital status
        X.loc[X['Marital_Status'] == 'Together', 'Marital_Status'] = 'Married'
        X.loc[(X['Marital_Status'] == 'Alone') | (X['Marital_Status'] == 'YOLO') | (X['Marital_Status'] == 'Absurd'), 'Marital_Status'] = 'Single'

        # Household size
        X['IsMarried'] = 0
        X.loc[X['Marital_Status'] == 'Married', 'IsMarried'] = 1
        X['HouseholdSize'] = 1 + X['Kidhome'] + X['Teenhome'] + X['IsMarried']

        # Total amount per capita
        X['MntTotal']   = X['MntWines'] + X['MntFruits'] + X['MntMeatProducts'] + X['MntFishProducts'] + X['MntSweetProducts'] + X['MntGoldProds']
        X['MntPerCapita'] = X['MntTotal'] / X['HouseholdSize']

        # Age
        X['Age'] = 2024 - X['Year_Birth']

        # Drop the unnecessary columns
        X.drop(columns=['Dt_Customer', 'EnrollmentYear', 'Kidhome', 'Teenhome', 'Marital_Status', 'MntTotal', 'Year_Birth', 'NumWebVisitsMonth','Complain'], inplace = True)

        return X
    
class Encode():
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        educations = ['Education_2n Cycle', 'Education_Basic', 'Education_Graduation', 'Education_Master', 'Education_PhD']

        X = pd.get_dummies(X, columns = ['Education'])

        for education in educations:
            if education not in X.columns:
                X[education] = False

        return X
    
class ScaleFeatures():
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        # Using MinMaxScaler
        scaler = ManualMinMaxScaler()

        # Features to scale
        mnt_features = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        purchases_features = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
        scaled_features = mnt_features + purchases_features + ['Recency', 'MntPerCapita', 'Age']

        # Fit to cleaned data
        scale = scaler.fit(pd.read_csv('data/cleaned_data.csv')[scaled_features])

        # Scale the features
        X[scaled_features] = scale.transform(X[scaled_features])

        return X