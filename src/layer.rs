use rand::Rng;
use crate::activation::Activation;

pub(crate) struct Layer {
    activation: Activation,
    num_inputs: usize,
    num_neurons: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,

    // Cache for the backpropagation
    last_inputs: Vec<f32>,
    last_sums: Vec<f32>,

    // These vectors contain the correction brought to each bias/wheight after the calculation of the error
    // in order to adapt these so the neural network provide a better result in output
    weight_gradients: Vec<f32>,
    bias_gradients: Vec<f32>,
}

impl Layer {
    pub fn new(num_inputs: usize, num_neurons: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();

        // Random initialization (Uniform distribution between -1 and 1)
        let weights = (0..num_inputs * num_neurons)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        // Biases can start at 0 without breaking symmetry
        let biases = vec![0.0; num_neurons];

        Self {
            num_inputs,
            num_neurons,
            weights,
            biases,
            last_inputs: vec![0.0; num_inputs],
            last_sums: vec![0.0; num_neurons],
            weight_gradients: vec![0.0; num_inputs * num_neurons],
            bias_gradients: vec![0.0; num_neurons],
            activation,
        }
    }
    // Forward propagation applied to this layer
    pub fn forward(&mut self, inputs: &[f32]) -> Vec<f32> {
        self.last_inputs = inputs.to_vec();
        let mut outputs = vec![0.0; self.num_neurons];

        for i in 0..self.num_neurons {
            let mut sum = self.biases[i];
            for j in 0..self.num_inputs {
                sum += inputs[j] * self.weights[i * self.num_inputs + j];
            }
            self.last_sums[i] = sum;
            outputs[i] = (self.activation.function)(sum);
        }
        outputs
    }
    // backpropagation applied to this layer
    pub fn backward(&mut self, next_layer_gradients: &[f32]) -> Vec<f32> {
        let mut prev_layer_gradients = vec![0.0; self.num_inputs];

        for i in 0..self.num_neurons {
            // local delta (dC/da * da/ds)
            let delta = next_layer_gradients[i] * (self.activation.derivative)(self.last_sums[i]);

            // Bias gradient
            self.bias_gradients[i] += delta;

            for j in 0..self.num_inputs {
                let idx = i * self.num_inputs + j;
                // Weight gradient (delta * input)
                self.weight_gradients[idx] += delta * self.last_inputs[j];

                // Prepare the error for the previous layer (sum of delta * weights)
                prev_layer_gradients[j] += delta * self.weights[idx];
            }
        }
        prev_layer_gradients
    }

    pub fn get_weights(&self) -> (&[f32], &[f32]) {
        (&self.weights, &self.biases)
    }

    pub fn set_weights(&mut self, weights: Vec<f32>, biases: Vec<f32>) {
        self.weights = weights;
        self.biases = biases;
    }

    pub fn update_params(&mut self, learning_rate: f32, batch_size: usize) {
        let n = batch_size as f32;

        for i in 0..self.weights.len() {
            // Apply the average of the accumulated gradient
            self.weights[i] -= learning_rate * (self.weight_gradients[i] / n);
            self.weight_gradients[i] = 0.0;
        }
        for i in 0..self.biases.len() {
            self.biases[i] -= learning_rate * (self.bias_gradients[i] / n);
            self.bias_gradients[i] = 0.0;
        }
    }
}
