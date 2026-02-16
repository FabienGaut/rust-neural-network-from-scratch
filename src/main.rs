mod activation;
mod layer;
mod network;
mod data;

use activation::Activation;
use network::NeuralNetwork;
use data::{load_csv_data, split_dataset, normalize_inputs, normalize_outputs, calculate_mse};

fn main() {
    let input_dim = 4;
    let output_dim = 1;
    let csv_path = "data/synthetic_data.csv";

    // Load and split the dataset
    let mut raw_data = load_csv_data(csv_path, input_dim, output_dim, false)
        .expect("Failed to load CSV");

    normalize_inputs(&mut raw_data);
    normalize_outputs(&mut raw_data);

    let (train_set, test_set) = split_dataset(raw_data, 0.8);

    // Initialize the neural network
    let shape = vec![input_dim, 16, 10, output_dim]; // 4 -> 16 -> 10 -> 1, the hidden layers are meant to be changed to test different formats of the neural network
    let activations = vec![Activation::leaky_relu(), Activation::leaky_relu(), Activation::identity()];
    let mut brain = NeuralNetwork::new(shape, activations);
    brain.learning_rate = 0.05;

    println!("Training on {} samples, Testing on {} samples...", train_set.len(), test_set.len());

    // Train the model on 1000 epoch
    for epoch in 0..=1000 {
        for batch in train_set.chunks(32) { // 32 is the default size of a batch here
            brain.train_on_batch(batch);
        }

        if epoch % 100 == 0 { // follow the evolution of the training
            let train_error = calculate_mse(&mut brain, &train_set);
            println!("Epoch {} | Train MSE: {:.6}", epoch, train_error);
        }
    }

    // Test the model
    let test_error = calculate_mse(&mut brain, &test_set);
    println!("\nFinal Test MSE: {:.6}", test_error);

    println!("\nSample Predictions:\n");
    for (input, target) in test_set.iter().take(5) {
        let pred = brain.predict(input);
        println!("Target: {:.4} | Predicted: {:.4}", target[0], pred[0]);
    }
}
