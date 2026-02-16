use std::error::Error;
use std::fs::File;
use rand::seq::SliceRandom;
use crate::network::NeuralNetwork;

// Divide the dataset into train and test sets to train and evaluate the training of it
pub fn split_dataset(
    dataset: Vec<(Vec<f32>, Vec<f32>)>,
    train_ratio: f32
) -> (Vec<(Vec<f32>, Vec<f32>)>, Vec<(Vec<f32>, Vec<f32>)>) {
    let mut rng = rand::thread_rng();
    let mut data = dataset;

    // shuffle the data before splitting the dataset to avoid biases
    data.shuffle(&mut rng);

    // define the train size
    let train_size = (data.len() as f32 * train_ratio) as usize;

    // split the data set
    let test_set = data.split_off(train_size);
    let train_set = data;

    (train_set, test_set)
}

// Load a csv file in a format that allows the model to read it properly
pub fn load_csv_data(path: &str, input_size: usize, output_size: usize, skip_index_0: bool) -> Result<Vec<(Vec<f32>, Vec<f32>)>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let mut dataset = Vec::new();

    for result in reader.records() {
        let record = result?;
        let row: Vec<f32> = if skip_index_0 {
            record.iter()
                .skip(1)
                .map(|s| s.parse::<f32>().unwrap_or(0.0))
                .collect()
        } else {
            record.iter()
                .map(|s| s.trim().parse::<f32>().unwrap_or(0.0))
                .collect()
        };

        if row.len() == input_size + output_size {
            let inputs = row[0..input_size].to_vec();
            let targets = row[input_size..input_size + output_size].to_vec();
            dataset.push((inputs, targets));
        }
    }

    println!("Loaded {} samples from {}", dataset.len(), path);
    Ok(dataset)
}

// Allows to get the distance between the prediction and the targetted output
pub fn calculate_mse(brain: &mut NeuralNetwork, dataset: &[(Vec<f32>, Vec<f32>)]) -> f32 {
    let mut total_error = 0.0;
    for (input, target) in dataset {
        let pred = brain.predict(input);
        total_error += (pred[0] - target[0]).powi(2);
    }
    total_error / dataset.len() as f32
}

// get standardized values in input/output in these functions
pub fn normalize_inputs(dataset: &mut Vec<(Vec<f32>, Vec<f32>)>) {
    if dataset.is_empty() { return; }

    let input_len = dataset[0].0.len();

    for i in 0..input_len {
        let min = dataset.iter().map(|(ins, _)| ins[i]).fold(f32::INFINITY, f32::min);
        let max = dataset.iter().map(|(ins, _)| ins[i]).fold(f32::NEG_INFINITY, f32::max);

        for (ins, _) in dataset.iter_mut() {
            if max != min {
                ins[i] = (ins[i] - min) / (max - min);
            }
        }
    }
}

pub fn normalize_outputs(dataset: &mut Vec<(Vec<f32>, Vec<f32>)>) {
    if dataset.is_empty() { return; }

    let output_len = dataset[0].1.len();

    for i in 0..output_len {
        let min = dataset.iter().map(|(_, outs)| outs[i]).fold(f32::INFINITY, f32::min);
        let max = dataset.iter().map(|(_, outs)| outs[i]).fold(f32::NEG_INFINITY, f32::max);

        for (_, outs) in dataset.iter_mut() {
            if max != min {
                outs[i] = (outs[i] - min) / (max - min);
            }
        }
    }
}
