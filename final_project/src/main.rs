/*controls the overall workflow through calling the functions to 
 train the random forest model, decision tree regressor, 
 and give an interactive function for inputting the patient’s features to give out the 
 recommended surgery type and the survival time after the surgery. 
 And it includes the tests for correct data loading, feature cleaning, and model training. 
 */
mod data;
mod model;
mod regression;

use data::{load_data, prepare_surgery_dataset, prepare_survival_dataset};
use model::{train_random_forest_classifier, evaluate_classifier};
use regression::{train_tree_regressor, evaluate_regressor};

use smartcore::model_selection::train_test_split;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::tree::decision_tree_regressor::DecisionTreeRegressor;
use smartcore::linalg::BaseMatrix;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "data/breast_cancer_data.csv";
    let records = load_data(file_path);

    let (features_surgery, labels_surgery, surgery_feature_names) = prepare_surgery_dataset(&records);
    let (features_survival, labels_survival, _survival_feature_names) = prepare_survival_dataset(&records);

    println!("\nDatasets Loaded:");
    println!("Surgery dataset: {} samples, {} features", features_surgery.shape().0, features_surgery.shape().1);
    println!("Survival dataset: {} samples, {} features", features_survival.shape().0, features_survival.shape().1);

    // Train/test split
    let (x_train_surgery, x_test_surgery, y_train_surgery, y_test_surgery) =
        train_test_split(&features_surgery, &labels_surgery, 0.8, true);

    let (x_train_survival, x_test_survival, y_train_survival, y_test_survival) =
        train_test_split(&features_survival, &labels_survival, 0.8, true);

    // Train models
    let classifier = train_random_forest_classifier(&x_train_surgery, &y_train_surgery);
    let regressor = train_tree_regressor(&x_train_survival, &y_train_survival);

    // Evaluate models
    let accuracy = evaluate_classifier(&classifier, &x_test_surgery, &y_test_surgery);
    let mse = evaluate_regressor(&regressor, &x_test_survival, &y_test_survival);

    println!("\nEvaluation Results:");
    println!("Surgery Accuracy: {:.4}", accuracy);
    println!("Survival MSE: {:.2}", mse);

    interactive_prediction(&classifier, &regressor, &surgery_feature_names)?;

    Ok(())
}


fn interactive_prediction( //Runs an interactive prompt where the user manually inputs a new patient's clinical data
    classifier: &smartcore::ensemble::random_forest_classifier::RandomForestClassifier<f64>,
    regressor: &DecisionTreeRegressor<f64>,
    feature_names: &[String],
) -> Result<(), Box<dyn std::error::Error>> {

    println!("New Patient Input");

    // binary fields that always take 0 or 1 (Yes/No)
    let binary_fields = vec![
        "chemotherapy",
        "hormone_therapy",
        "radio_therapy",
        "er_status",
        "pr_status",
        "her2_status",
        "death_from_cancer",
        "received_targeted_therapy",
    ];

    loop {
        // store numeric feature values for prediction
        let mut surgery_features = Vec::new();
        let mut survival_features = Vec::new();

        // Prompt for the user for answering each feature
        for feature in feature_names {
            let feature_lower = feature.to_lowercase();
            println!("Answer for: {}", feature);

            // Show helpful options based on field type
            if feature_lower.contains("cellularity") {
                println!("Options: 0 = Low, 1 = Moderate, 2 = High");
            } else if feature_lower.contains("primary_tumor_laterality") {
                println!("Options: 0 = Right, 1 = Left, 2 = Other");
            } else if feature_lower.contains("nottingham_prognostic_index") {
                println!("Typical Range: 1.0 - 6.0");
            } else if feature_lower.contains("neoplasm_histologic_grade") {
                println!("Options: 1, 2, 3");
            } else if feature_lower.contains("cancer_type") {
                println!("Options: 0 = Ductal, 1 = Lobular, 2 = Mixed");
            } else if feature_lower.contains("pam50") {
                println!("Options: 0 = LumA, 1 = LumB, 2 = HER2, 3 = Basal, 4 = Normal, 5 = Claudin-low");
            } else if feature_lower.contains("menopausal") {
                println!("Options: 0 = Pre, 1 = Post");
            } else if feature_lower.contains("tumor_stage") {
                println!("Options: 1 = Stage I, 2 = Stage II, 3 = Stage III, 4 = Stage IV");
            } else if binary_fields.contains(&feature_lower.as_str()) {
                println!("Options: 0 = No, 1 = Yes");
            } else if feature_lower.contains("age") {
                println!("Typical Range: 20-90 years old");
            } else if feature_lower.contains("tumor_size") {
                println!("Typical Range: 5mm - 100mm");
            } else if feature_lower.contains("mutation_count") {
                println!("Typical Range: 0-100 mutations");
            } else if feature_lower.contains("lymph_nodes_examined_positive") {
                println!("Typical Range: 0-45 nodes");
            } else if feature_lower.contains("integrative_cluster") {
                println!("Options: 1-10");
            } else {
                println!("(Enter a numeric value)");
            }

            // read user input
            print!("Input for {}: ", feature);
            io::stdout().flush()?;  // Make sure the prompt is shown before reading input
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let value = input.trim().parse::<f64>().unwrap_or(0.0); 

            surgery_features.push(value);
            survival_features.push(value);
        }

        // Warn if all features are 0 (likely the user entered invalid data)
        if surgery_features.iter().all(|&v| v == 0.0) {
            println!("\nWarning: All entries were zero. Please re-enter real patient information!");
            continue;  // Loop again
        }

        // STEP 1: Predict Surgery Type (Classification)
        let x_surgery = DenseMatrix::from_array(1, surgery_features.len(), &surgery_features);
        let surgery_prediction = classifier.predict(&x_surgery)?[0];
        let recommended = if surgery_prediction == 0.0 { "Breast Conserving" } else { "Mastectomy" };
        println!("\nRecommended Surgery Type: {}", recommended);

        // STEP 2: Ask the user for the actual surgery the patient had (adds to survival feature set)
        println!("Which surgery did the patient actually receive? (0 = Breast Conserving, 1 = Mastectomy)");
        let mut surgery_input = String::new();
        io::stdin().read_line(&mut surgery_input)?;
        let actual_surgery = surgery_input.trim().parse::<f64>().unwrap_or(0.0);
        survival_features.push(actual_surgery);  // Important to include actual surgery in survival prediction

        // STEP 3: Predict Survival Months (Regression)
        let x_survival = DenseMatrix::from_array(1, survival_features.len(), &survival_features);
        let pred_ln = regressor.predict(&x_survival)?[0];  // predict log-survival
        let survival_prediction = pred_ln.exp();  // back-transform to original scale

        println!("\nPredicted Overall Survival: {:.1} months", survival_prediction.max(1.0));
        break;  // Exit if get a complete data input
    }

    Ok(())
}



#[cfg(test)]
mod tests {
    use crate::data::{load_data, prepare_surgery_dataset, prepare_survival_dataset};
    use crate::model::{train_random_forest_classifier, evaluate_classifier};
    use crate::regression::{train_tree_regressor, evaluate_regressor};
    use smartcore::model_selection::train_test_split;
    use smartcore::linalg::BaseMatrix;

    #[test]
    fn test_data_loading() {
        let file_path = "data/breast_cancer_data.csv";
        let records = load_data(file_path);
        assert!(!records.is_empty(), "Dataset should not be empty");
    }

    #[test]
    fn test_prepare_surgery_dataset() {
        let file_path = "data/breast_cancer_data.csv";
        let records = load_data(file_path);
        let (features, labels, feature_names) = prepare_surgery_dataset(&records);
        assert_eq!(features.shape().0, labels.len(), "Features and labels must have matching samples");
    }

    #[test]
    fn test_prepare_survival_dataset() {
        let file_path = "data/breast_cancer_data.csv";
        let records = load_data(file_path);
        let (features, labels, feature_names) = prepare_survival_dataset(&records);
        assert_eq!(features.shape().0, labels.len(), "Features and labels must have matching samples");
        assert!(feature_names.len() > 0, "Should have extracted survival feature names");
    }

    #[test]
    fn test_model_training_and_prediction() {
        let file_path = "data/breast_cancer_data.csv";
        let records = load_data(file_path);

        let (features_surgery, labels_surgery, _) = prepare_surgery_dataset(&records);
        let (features_survival, labels_survival, _) = prepare_survival_dataset(&records);

        let (x_train_surgery, x_test_surgery, y_train_surgery, y_test_surgery) =
            train_test_split(&features_surgery, &labels_surgery, 0.8, true);

        let (x_train_survival, x_test_survival, y_train_survival, y_test_survival) =
            train_test_split(&features_survival, &labels_survival, 0.8, true);

        let classifier = train_random_forest_classifier(&x_train_surgery, &y_train_surgery);
        let regressor = train_tree_regressor(&x_train_survival, &y_train_survival);

        let acc = evaluate_classifier(&classifier, &x_test_surgery, &y_test_surgery);
        let mse = evaluate_regressor(&regressor, &x_test_survival, &y_test_survival);

        assert!(acc >= 0.0 && acc <= 1.0, "Accuracy should be between 0 and 1");
        assert!(mse >= 0.0, "MSE should be non-negative");
    }
}
