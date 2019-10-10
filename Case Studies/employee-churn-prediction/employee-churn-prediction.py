import pandas as pd
from h2o import h2o

# Removing existing data from H2O Cluster

h2o.init(ip="localhost", port=54321)
h2o.remove_all()

# Loading HR Analytics Data from CSV File
full_data_frame = h2o.H2OFrame(pd.read_csv("dataset/HR_comma_sep.csv", index_col=None, header=0))

# Defining categorical features
feature_columns = [
    'left',
    'Work_accident',
    'promotion_last_5years',
    'department'
]

# Defining continuous features
continuous_feature_columns = [
    'satisfaction_level',
    'last_evaluation',
    'number_project',
    'average_montly_hours',
    'time_spend_company',
    'salary'
]

training_data_frame, test_data_frame = full_data_frame.split_frame(ratios=[.8])
training_data_frame[feature_columns] = training_data_frame[feature_columns].asfactor()
test_data_frame[feature_columns] = test_data_frame[feature_columns].asfactor()

print(training_data_frame[0, :])
print(test_data_frame[0, :])

feature_columns.extend(continuous_feature_columns)
training_data_frame = training_data_frame[feature_columns]
test_data_frame = test_data_frame[feature_columns]

# Initialize Random Forest Estimator
random_forest_model = h2o.H2ORandomForestEstimator(
    model_id="HREmployeeChurnPredictionRandomForest",
    ntrees=10,
    max_depth=10,
    min_rows=4,
    nfolds=10,
    seed=12345
)

random_forest_model.train(y='left', training_frame=training_data_frame)
random_forest_model_performance = random_forest_model.model_performance(test_data=test_data_frame)
print(random_forest_model_performance)

# Initialize Naive Bayes Estimator
naive_bayes_model = h2o.H2ONaiveBayesEstimator(
    model_id="HREmployeeChurnPredictionNaiveBayes",
    nfolds=10,
    seed=123456
)

naive_bayes_model.train(y='left', training_frame=training_data_frame)
naive_bayes_model_performance = naive_bayes_model.model_performance(test_data=test_data_frame)
print(naive_bayes_model_performance)

# Initialize Gradient Boosting Estimator
gradient_boosting_model = h2o.H2OGradientBoostingEstimator(
    model_id="HREmployeeChurnPredictionGradientBoosting",
    ntrees=10,
    max_depth=10,
    stopping_tolerance=0.01,
    stopping_rounds=2,
    score_each_iteration=True,
    seed=1234567
)

gradient_boosting_model.train(y='left', training_frame=training_data_frame)
gradient_boosting_model_performance = gradient_boosting_model.model_performance(test_data=test_data_frame)
print(gradient_boosting_model)

# Initialize Generalized Linear Estimator
generalized_linear_model = h2o.H2OGeneralizedLinearEstimator(
    model_id="HREmployeeChurnPredictionGeneralizedLinear",
    nfolds=10,
    max_iterations=20,
    family="binomial",
    seed=1234567
)
generalized_linear_model.train(y='left', training_frame=training_data_frame)
generalized_linear_model_performance = generalized_linear_model.model_performance(test_data=test_data_frame)
print(generalized_linear_model_performance)