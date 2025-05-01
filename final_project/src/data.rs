//load patient data and prepare numerical and textual labels for training models
use serde::Deserialize;  
use smartcore::linalg::naive::dense_matrix::DenseMatrix;  
use serde_json::Value;  // hold flexible data types (numbers, strings, null)
use std::collections::HashMap;  
use std::fs::File;  
use std::io::BufReader;  // buffered file reading
use csv::ReaderBuilder;  

/// Struct representing one row of the dataset
#[derive(Debug, Deserialize)]
pub struct Record {
    #[serde(flatten)]  // Flatten the CSV row into a HashMap
    pub fields: HashMap<String, Value>,
}

/// Load the CSV data into a Vec<Record>
pub fn load_data(file_path: &str) -> Vec<Record> {
    let file = File::open(file_path).expect("Cannot open file");
    let mut rdr = ReaderBuilder::new()
        .has_headers(true) 
        .from_reader(BufReader::new(file));

    let mut records = Vec::new();  // store all rows
    for result in rdr.deserialize() {
        let record: Record = result.expect("Cannot deserialize record");  // Deserialize each row
        records.push(record);
    }
    records
}

/// Identify columns that have too many missing values
pub fn identify_noisy_columns(records: &[Record], missing_threshold: f64) -> Vec<String> { 
    //Input: &[Record], missing data precentage threshold,  Output: column names that exceed threshold
    if records.is_empty() {
        return vec![];  // No records, return empty vec
    }
    let total = records.len() as f64;
    let mut missing_counts: HashMap<String, usize> = HashMap::new();  // Keep track of missing counts per column
    // Count missing values per column
    for record in records {
        for (key, val) in &record.fields {
            if val.is_null() || matches!(val, Value::String(s) if s.trim().is_empty()) {
                *missing_counts.entry(key.to_string()).or_insert(0) += 1;
            }
        }
    }
    // Filter out columns where missing data ratio exceeds threshold
    missing_counts
        .into_iter()
        .filter(|(_, count)| *count as f64 / total > missing_threshold) //use iterator + closure
        .map(|(key, _)| key) //again, use iterator + closure
        .collect() 
}
// Encode categorical string data into numeric form (very hardcoding, but it is worthy because I need to make it easiser for the interactive part)
fn encode_string(field: &str, val: &str) -> f64 {
    //input: field name & value, output: numeric encoding
    match field.to_lowercase().as_str() {
        "cellularity" => match val.to_lowercase().as_str() {
            "low" => 0.0, "moderate" => 1.0, "high" => 2.0, _ => 0.0
        },
        "pam50_subtype" => match val.to_lowercase().as_str() {
            "luma" => 0.0, "lumb" => 1.0, "her2" => 2.0, "basal" => 3.0,
            "normal" => 4.0, "claudin-low" => 5.0, _ => 6.0
        },
        "er_status" | "pr_status" | "her2_status" => match val.to_lowercase().as_str() {
            "positive" => 1.0, "negative" => 0.0, _ => 0.0
        },
        "inferred_menopausal_state" => match val.to_lowercase().as_str() {
            "pre" => 0.0, "post" => 1.0, _ => 0.0
        },
        "primary_tumor_laterality" => match val.to_lowercase().as_str() {
            "right" => 0.0, "left" => 1.0, "other" => 2.0, _ => 2.0
        },
        "cancer_type_detailed" => match val.to_lowercase().as_str() {
            "ductal carcinoma" => 0.0, "lobular carcinoma" => 1.0, "mixed" => 2.0, _ => 2.0
        },
        "type_of_breast_surgery" => match val.to_lowercase().as_str() {
            "breast conserving" => 0.0, "mastectomy" => 1.0, _ => 1.0
        },
        _ => 0.0,  // Default for unknown categories
    }
}

/// Apply transformations
fn transform_value(field: &str, value: f64) -> f64 {
    match field.to_lowercase().as_str() {
        "tumor_size" | "lymph_nodes_examined_positive" => (value + 1.0).ln(),  // Log transform 
        _ => value,
    }
}

/// Define a list of features we want to include in the dataset, I set the lifetime as static
fn selected_features() -> Vec<&'static str> {
    vec![
        "age_at_diagnosis", "chemotherapy", "neoplasm_histologic_grade", "hormone_therapy",
        "lymph_nodes_examined_positive", "nottingham_prognostic_index", "radio_therapy",
        "tumor_size", "tumor_stage", "type_of_breast_surgery", "cancer_type_detailed",
        "cellularity", "pam50_subtype", "er_status", "her2_status", "inferred_menopausal_state",
        "integrative_cluster", "primary_tumor_laterality", "pr_status"
    ]
}

/// Build dataset for Surgery Recommendation (classification task)
pub fn prepare_surgery_dataset(records: &[Record]) -> (DenseMatrix<f64>, Vec<f64>, Vec<String>) {
    //Input: Records, Output: Features matrix, Labels for suregry type, and feature names
    let noisy_cols = identify_noisy_columns(records, 0.2);  // Drop columns with >20% missing
    let feature_keys: Vec<String> = selected_features()
        .into_iter()
        .filter(|k| k != &"type_of_breast_surgery" && !noisy_cols.contains(&k.to_string()))//iterator + closure
        .map(|s| s.to_string()) //iterartor + closure
        .collect();

    let mut features = Vec::new();
    let mut labels = Vec::new();

    // Process each record
    for record in records {
        let mut row = Vec::new();
        for feature in &feature_keys {
            let val = record.fields.get(feature);
            let v = match val {
                Some(Value::Number(n)) => transform_value(feature, n.as_f64().unwrap_or(0.0)),
                Some(Value::String(s)) => encode_string(feature, s),
                _ => 0.0,
            };
            row.push(v);
        }

        // Process the label (surgery type)
        if let Some(label_val) = record.fields.get("type_of_breast_surgery") {
            let y = match label_val {
                Value::Number(num) => num.as_f64().unwrap_or(1.0),
                Value::String(s) => encode_string("type_of_breast_surgery", s),
                _ => 1.0,
            };
            features.push(row);
            labels.push(y);
        }
    }

    // Flatten the features into a 1D array for DenseMatrix (get from ChatGPT)
    let flat = features.iter().flatten().copied().collect::<Vec<f64>>();
    let matrix = DenseMatrix::from_array(features.len(), feature_keys.len(), &flat);

    // For easy reading
    println!("\nSurgery Dataset:");
    println!("Records: {}", features.len());
    println!("Dropped noisy columns: {:?}", noisy_cols);
    println!("Features per record: {}", feature_keys.len());

    (matrix, labels, feature_keys)
}

// Build dataset for Survival Prediction (regression task) - literally the same things as last one, but just add survival month
pub fn prepare_survival_dataset(records: &[Record]) -> (DenseMatrix<f64>, Vec<f64>, Vec<String>) {
    //Input: Records, Output: feature matrix, labels for survival month, and feature names
    let noisy_cols = identify_noisy_columns(records, 0.2);
    let feature_keys: Vec<String> = selected_features()
        .into_iter()
        .filter(|k| !noisy_cols.contains(&k.to_string()))
        .map(|s| s.to_string())
        .collect();

    let mut features = Vec::new();
    let mut labels = Vec::new();

    for record in records {
        let mut row = Vec::new();
        for feature in &feature_keys {
            let val = record.fields.get(feature);
            let v = match val {
                Some(Value::Number(n)) => transform_value(feature, n.as_f64().unwrap_or(0.0)),
                Some(Value::String(s)) => encode_string(feature, s),
                _ => 0.0,
            };
            row.push(v);
        }

        // Process the label (overall_survival_months)
        if let Some(label_val) = record.fields.get("overall_survival_months") {
            let y = match label_val {
                Value::Number(num) => num.as_f64().unwrap_or(0.0),
                Value::String(s) => s.parse::<f64>().unwrap_or(0.0),
                _ => 0.0,
            };

            if y > 0.0 { // Only include positive survival months (just in case)
                features.push(row);
                labels.push(y.ln()); 
            }
        }
    }

    let flat = features.iter().flatten().copied().collect::<Vec<f64>>();
    let matrix = DenseMatrix::from_array(features.len(), feature_keys.len(), &flat);

    println!("\nSurvival Dataset:");
    println!("Records: {}", features.len());
    println!("Dropped noisy columns: {:?}", noisy_cols);
    println!("Features per record: {}", feature_keys.len());

    (matrix, labels, feature_keys)
}
