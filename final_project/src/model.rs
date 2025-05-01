//use training data and apply to Random Forest model for predicting suitable surgery type
use smartcore::ensemble::random_forest_classifier::{RandomForestClassifier, RandomForestClassifierParameters};
use smartcore::metrics::accuracy;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;


// What it does: Trains a Random Forest model using the provided training data 
pub fn train_random_forest_classifier(
    x_train: &DenseMatrix<f64>, // Feature matrix (each row = one sample)
    y_train: &Vec<f64>,//Labels for each sample.
) -> RandomForestClassifier<f64> { //Returns a trained RandomForestClassifier<f64>
    // Fit the RandomForest model
    RandomForestClassifier::fit(
        x_train,
        y_train,
        RandomForestClassifierParameters {
            n_trees: 100,               // Use 100 trees in the ensemble
            max_depth: Some(10),        // Limit each tree to depth 10
            ..Default::default()        // Use default values for the rest
        }
    ).expect("Failed to train Random Forest model")
}


// What it does: Tests the trained model using test data and reports accuracy.
pub fn evaluate_classifier(
    model: &RandomForestClassifier<f64>, //Trained model
    x_test: &DenseMatrix<f64>,//test feature data
    y_test: &Vec<f64>,//test labels
) -> f64 {//returns accuracy
    // Predict the labels for test data.
    let predictions = model.predict(x_test).expect("Failed to predict on test set");
    
    // use accuracy() from Smartcore to compare predicted vs actual labels.
    accuracy(y_test, &predictions)
}
