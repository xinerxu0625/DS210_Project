//trains and evaluates a Decision Tree Regressor for predicting survival months after having surgery
use smartcore::linalg::naive::dense_matrix::DenseMatrix;  
use smartcore::tree::decision_tree_regressor::{DecisionTreeRegressor, DecisionTreeRegressorParameters};  // Decision tree regression
use smartcore::metrics::mean_squared_error;  

pub fn train_tree_regressor( //Trains a Decision Tree Regressor on the training data to predict survival months.
    x_train: &DenseMatrix<f64>, //training features
    y_train: &Vec<f64>, //training labels
) -> DecisionTreeRegressor<f64> {
    // Fit the decision tree regressor with specified hyperparameters
    DecisionTreeRegressor::fit(
        x_train,
        y_train,
        DecisionTreeRegressorParameters {
            max_depth: Some(10),  // Limit tree depth to prevent overfitting
            min_samples_split: 2,  // Minimum number of samples needed to split a node
            min_samples_leaf: 1,   // Minimum samples per leaf
            ..Default::default()  // Other settings are default
        },
    ).expect("Failed to train Decision Tree Regressor")
}


/// - Mean Squared Error (f64): lower is better
pub fn evaluate_regressor( // Evaluates the trained regressor by computing Mean Squared Error (MSE)
    model: &DecisionTreeRegressor<f64>,//test features
    x_test: &DenseMatrix<f64>,//true labels
    y_test: &Vec<f64>,
) -> f64 {
    // Predict the survival months using the test set
    let predictions = model.predict(x_test).expect("Prediction failed");
    // Compute Mean Squared Error between true labels and predictions
    mean_squared_error(y_test, &predictions)
}
