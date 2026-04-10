mod activation;
mod layer;
mod network;
mod data;
mod configparser;

use configparser::NetworkConfig;
use network::NeuralNetwork;
use data::{load_csv_data, split_dataset, normalize_inputs, normalize_outputs, calculate_mse};

fn main() {
    let config = NetworkConfig::from_file("config/config.ini");

    let mut raw_data = load_csv_data(
        &config.csv_path,
        config.input_dim,
        config.output_dim,
        config.skip_index_0,
    ).expect("Failed to load CSV");

    normalize_inputs(&mut raw_data);
    normalize_outputs(&mut raw_data);

    let (train_set, test_set) = split_dataset(raw_data, 1.0 - config.test_split);

    let shape = config.shape();
    let activations = config.activations;

    println!("Network shape: {:?}", shape);

    let mut brain = NeuralNetwork::new(shape, activations);
    brain.learning_rate = config.learning_rate;

    println!("Training on {} samples, Testing on {} samples...", train_set.len(), test_set.len());

    for epoch in 0..=config.epochs {
        for batch in train_set.chunks(config.batch_size) {
            brain.train_on_batch(batch);
        }

        if epoch % 100 == 0 {
            let train_error = calculate_mse(&mut brain, &train_set);
            println!("Epoch {} | Train MSE: {:.6}", epoch, train_error);
        }
    }

    let test_error = calculate_mse(&mut brain, &test_set);
    println!("\nFinal Test MSE: {:.6}", test_error);

    println!("\nSample Predictions:\n");
    for (input, target) in test_set.iter().take(5) {
        let pred = brain.predict(input);
        println!("Target: {:.4} | Predicted: {:.4}", target[0], pred[0]);
    }
}
