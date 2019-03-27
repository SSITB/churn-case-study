import pandas as pd

class DataSelector:
    def __init__(self):
        self.raw_data = self
        self.today = pd.to_datetime('7/1/2014')
        self.churn_thresh = pd.Timedelta(30, 'D')
    
    def clean_data_gradient_boost(self, df):
        # send data to staging for initialization and separation
        X, y = self.stage_cleaning(df)

        # clean x and y
        X = self.cleanX_gradient_boost(X)
        y = self.cleany(y)
        return X, y
        
    def stage_cleaning(self, df):
        # stage replacing null values with the median
        self.rider_null = df['avg_rating_by_driver'].median()
        self.driver_null = df['avg_rating_of_driver'].median()

        # separate x and y
        X = df.drop('last_trip_date', axis=1)
        y = pd.DataFrame(df['last_trip_date'])
        return X, y
    
    def cleany(self, y):
        
        # Defining churn: equal to 1 if last trip date happened more than 30 days ago
        y.loc[y['last_trip_date'] >= self.today - self.churn_thresh, 'churn'] = 0  # did not churn
        y.loc[y['last_trip_date'] < self.today - self.churn_thresh, 'churn'] = 1  # did churn
        return y['churn']
    
    def cleanX_gradient_boost(self, X):

        # Replacing null values with the median   
        X['avg_rating_by_driver']=X['avg_rating_by_driver'].apply(
                                        lambda x: self.rider_null if pd.isnull(x) else x)
        X['avg_rating_of_driver']=X['avg_rating_of_driver'].apply(
                                        lambda x: self.driver_null if pd.isnull(x) else x)

        # Creating dummies
        X['iPhone_user'] = pd.get_dummies(
                    X['phone'].map({'iPhone':'iPhone'}))
        X['luxury_car_user'] = pd.get_dummies(
                    X['luxury_car_user'].map({True:1}))
        X[['Astapor','Winterfel']]=pd.get_dummies(
                    X['city'].map({'Astapor':'Astapor','Winterfell':'Winterfell'}))
        X['weekend_signup']=X['signup_date'].apply(lambda x: x.weekday()>4)
        X['weekend_signup'] = pd.get_dummies(
                    X['weekend_signup'].map({True:1}))

        #Defining features matrix
        self.finalX=X[['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver', 'avg_surge',
              'surge_pct', 'trips_in_first_30_days', 'luxury_car_user', 'weekday_pct',
              'iPhone_user', 'Astapor', 'Winterfel','weekend_signup']]

        return self.finalX
    
    
    #for logit we further categorized the highly skewed continuous variables
    def clean_data_logit(self, df):
        # send data to staging for initialization and separation
        X, y = self.stage_cleaning(df)

        # clean x and y
        X = self.cleanX_logit(X)
        y = self.cleany(y)
        return X, y
    
    
    def cleanX_logit(self, X):
        
        X=self.cleanX_gradient_boost(X)
        
        #The percent of trips taken with surge multiplier > 1, equal to 1 if that pct is above 0
        X.loc[:,'surge_above_zero_dummy']=X['surge_pct'].where(X['surge_pct']==0,1) 
    
        #Rider rating 
        X.loc[:,'avg_rating_by_driver']=X['avg_rating_by_driver'].where(
                X['avg_rating_by_driver']>=4,0)
        X.loc[:,'avg_rating_by_driver']=X['avg_rating_by_driver'].where(
                (X['avg_rating_by_driver']==5) | (X['avg_rating_by_driver']==0),1)
        X.loc[:,'avg_rating_by_driver']=X['avg_rating_by_driver'].where(
                X['avg_rating_by_driver']<5,2)
    
        X[['rider_rtg_4_5','rider_rtg_5']] = pd.get_dummies(
                X['avg_rating_by_driver'].map({1:'rider_rtg_4_5', 2:'rider_rtg_5'}))
        
        #Driver rating
        X.loc[:,'avg_rating_of_driver']=X['avg_rating_of_driver'].where(
                X['avg_rating_of_driver']>=4,0)
        X.loc[:,'avg_rating_of_driver']=X['avg_rating_of_driver'].where(
                (X['avg_rating_of_driver']==5) | (X['avg_rating_of_driver']==0),1)
        X.loc[:,'avg_rating_of_driver']=X['avg_rating_of_driver'].where(
                X['avg_rating_of_driver']<5,2)

        X[['driver_rtg_4_5','driver_rtg_5']] = pd.get_dummies(
                X['avg_rating_of_driver'].map({1:'driver_rtg_4_5', 2:'driver_rtg_5'}))

        #Weekday pct
        X.loc[:,'weekday_pct']=X['weekday_pct'].where(
                (X['weekday_pct']==100) | (X['weekday_pct']==0),1)
        X.loc[:,'weekday_pct']=X['weekday_pct'].where(X['weekday_pct']<100,2)
    
        X[['weekday_pct_above0_below100','weekday_pct_100']] = pd.get_dummies(
                X['weekday_pct'].map({1:'weekday_pct_above0_below100', 2:'weekday_pct_100'}))

        #Average distance
        X.loc[:,'avg_dist']=X['avg_dist'].where(X['avg_dist']>=20,0)
        X.loc[:,'avg_dist_above20miles']=X['avg_dist'].where((X['avg_dist']<20),1)
    
        #Trips in the first 30 days, equal to 1 if 10 or more trips in last 30 days 
        X.loc[:,'trips_in_first_30_days']=X['trips_in_first_30_days'].where(
                                            X['trips_in_first_30_days']>=10,0)
        X.loc[:,'first_30d_10or_more']=X['trips_in_first_30_days'].where(
                                            X['trips_in_first_30_days']<10,1)
        
        X=X.drop(['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver', 
                  'surge_pct', 'trips_in_first_30_days', 'weekday_pct',
                  'avg_surge','rider_rtg_5','driver_rtg_5', 'weekend_signup'],axis=1)
    
        return X
    