use std::collections::HashMap;
use std::fs;
use crate::activation::Activation;

pub struct NetworkConfig {
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub test_split: f32,
    pub csv_path: String,
    pub input_dim: usize,
    pub output_dim: usize,
    pub skip_index_0: bool,
    pub hidden_sizes: Vec<usize>,
    pub activations: Vec<Activation>,
}

impl NetworkConfig {
    pub fn from_file(path: &str) -> Self {
        let raw = fs::read_to_string(path)
            .unwrap_or_else(|_| panic!("Cannot read config file: {}", path));

        let sections = parse_ini(&raw);

        let training = sections.get("training").expect("Missing [training] section");
        let data = sections.get("data").expect("Missing [data] section");
        let layers = sections.get("layers").expect("Missing [layers] section");

        let input_dim = get_usize(data, "input_dim");
        let output_dim = get_usize(data, "output_dim");

        // Parse hidden layers: find all hidden_N_size keys, sorted by N
        let mut layer_indices: Vec<usize> = layers.keys()
            .filter_map(|k| {
                k.strip_prefix("hidden_")
                    .and_then(|rest| rest.strip_suffix("_size"))
                    .and_then(|n| n.parse::<usize>().ok())
            })
            .collect();
        layer_indices.sort();

        let mut hidden_sizes = Vec::new();
        let mut activations = Vec::new();

        for idx in &layer_indices {
            let size_key = format!("hidden_{}_size", idx);
            let act_key = format!("hidden_{}_activation", idx);

            let size = get_usize(layers, &size_key);
            let act_name = get_str(layers, &act_key);
            hidden_sizes.push(size);
            activations.push(parse_activation(&act_name));
        }

        // Output activation
        let output_act = get_str(layers, "output_activation");
        activations.push(parse_activation(&output_act));

        NetworkConfig {
            batch_size: get_usize(training, "batch_size"),
            epochs: get_usize(training, "epochs"),
            learning_rate: get_f32(training, "learning_rate"),
            test_split: get_f32(training, "test_split"),
            csv_path: get_str(data, "csv_path"),
            input_dim,
            output_dim,
            skip_index_0: get_str(data, "skip_index_0") == "true",
            hidden_sizes,
            activations,
        }
    }

    /// Build the shape vector: [input_dim, hidden_1, hidden_2, ..., output_dim]
    pub fn shape(&self) -> Vec<usize> {
        let mut shape = vec![self.input_dim];
        shape.extend_from_slice(&self.hidden_sizes);
        shape.push(self.output_dim);
        shape
    }
}

// --- Simple INI parser ---

fn parse_ini(content: &str) -> HashMap<String, HashMap<String, String>> {
    let mut sections: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut current_section = String::new();

    for line in content.lines() {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line.starts_with('[') && line.ends_with(']') {
            current_section = line[1..line.len() - 1].to_string();
            sections.entry(current_section.clone()).or_default();
        } else if let Some((key, value)) = line.split_once('=') {
            let key = key.trim().to_string();
            // Strip inline comments
            let value = value.split('#').next().unwrap_or("").trim().to_string();
            if !current_section.is_empty() {
                sections.get_mut(&current_section).unwrap().insert(key, value);
            }
        }
    }

    sections
}

fn get_str(section: &HashMap<String, String>, key: &str) -> String {
    section.get(key)
        .unwrap_or_else(|| panic!("Missing key: {}", key))
        .clone()
}

fn get_usize(section: &HashMap<String, String>, key: &str) -> usize {
    get_str(section, key).parse()
        .unwrap_or_else(|_| panic!("Invalid usize for key: {}", key))
}

fn get_f32(section: &HashMap<String, String>, key: &str) -> f32 {
    get_str(section, key).parse()
        .unwrap_or_else(|_| panic!("Invalid f32 for key: {}", key))
}

fn parse_activation(name: &str) -> Activation {
    match name {
        "sigmoid" => Activation::sigmoid(),
        "relu" => Activation::relu(),
        "leaky_relu" => Activation::leaky_relu(),
        "tanh" => Activation::tanh(),
        "identity" => Activation::identity(),
        _ => panic!("Unknown activation function: {}", name),
    }
}
