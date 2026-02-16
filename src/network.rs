use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use serde::{Serialize, Deserialize};
use crate::activation::Activation;
use crate::layer::Layer;

#[derive(Serialize, Deserialize)]
struct LayerData {
    weights: Vec<f32>,
    biases: Vec<f32>,
}

pub struct NeuralNetwork {
    layers: Vec<Layer>,
    pub learning_rate: f32,
}

impl NeuralNetwork {
    pub fn new(shape: Vec<usize>, activations: Vec<Activation>) -> Self {
        if shape.len() < 2 { panic!("At least one input and one output are required"); }
        if activations.len() != shape.len() - 1 {
            panic!("The number of activations must match the number of intervals between layers");
        }

        let mut layers = Vec::new();

        for i in 0..shape.len() - 1 {
            layers.push(Layer::new(
                shape[i],
                shape[i+1],
                activations[i]
            ));
        }

        Self {
            layers,
            learning_rate: 0.01,
        }
    }

    // We train the model with the use of batchs in order to optimize the training
    pub fn train_on_batch(&mut self, batch: &[(Vec<f32>, Vec<f32>)]) {
        // iterate over the entire batch
        for (input, target) in batch {
            let output = self.predict(input);

            // Compute the initial error
            let mut gradients: Vec<f32> = output.iter().zip(target.iter())
                .map(|(p, t)| p - t).collect();

            // backpropagation (accumulates gradients without updating weights)
            for layer in self.layers.iter_mut().rev() {
                gradients = layer.backward(&gradients);
            }
        }

        // apply the average once at the end of the batch
        for layer in &mut self.layers {
            layer.update_params(self.learning_rate, batch.len());
        }
    }

    // Predict an output result from inputs provided
    pub fn predict(&mut self, input: &[f32]) -> Vec<f32> {
        let mut current = input.to_vec();
        for layer in &mut self.layers {
            current = layer.forward(&current);
        }
        current
    }
}
