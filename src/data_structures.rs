use rand::{thread_rng, Rng, distributions::Uniform};

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn forward(&mut self, inputs: &Vec<f64>) {
        let mut inp = inputs.clone();
        for layer in self.layers.iter_mut() {
            layer.forward(&inp);
            inp = layer.activation.output();
        }
    }
}

pub struct Layer {
    pub neurons: LayerDense,
    pub activation: Activations,
}

impl Layer {
    pub fn forward(&mut self, inputs: &Vec<f64>) {
        self.neurons.forward(inputs);
        self.activation.forward(&self.neurons.output);
    }
}

pub struct LayerDense {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub inputs: Vec<f64>,
    pub output: Vec<f64>,
    pub dinputs: Vec<f64>,
    pub dweights: Vec<Vec<f64>>,
    pub dbiases: Vec<f64>,
}

impl LayerDense {
    pub fn new(n_inputs: u64, n_neurons: u64) -> Self {
        let mut rng = thread_rng();
        let range = Uniform::new(-1.0, 1.0);
        let weights: Vec<Vec<f64>> = (0..n_neurons).map(|_| (0..n_inputs).map(|_| rng.sample(&range)).collect()).collect();
        let biases: Vec<f64> = (0..n_neurons).map(|_| 0.0).collect();
        
        Self {
            weights,
            biases,
            inputs: vec![],
            output: vec![],
            dinputs: vec![],
            dweights: vec![],
            dbiases: vec![],
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) {
        self.inputs = inputs.to_owned();
        self.output = layer_outputs(inputs, &self.weights, &self.biases);
    }

    pub fn backward(&mut self, dvalues: &Vec<f64>) {
        self.dinputs = dot_matrix_vector(dvalues, &self.weights);
        self.dweights = dvalues.iter().map(|dvalue| self.inputs.iter().map(|input| dvalue * input).collect()).collect();
        self.dbiases = dvalues.clone();
    }
}

pub struct Accuracy {
}

impl Accuracy {
    pub fn calculate(output: &Vec<f64>, index: usize) -> f64 {
        let arg_max = output.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index).unwrap();
        (arg_max == index) as u8 as f64
    }
}

pub struct Loss {
    dinputs: Vec<f64>,
}

impl Loss {
    pub fn forward(output: &Vec<f64>, index: usize) -> f64 {
        // clipping to prevent exploding
        let prediction = output[index].max(0.00000001).min(1.0-0.00000001);
        let loss = -(prediction.ln());
        loss
    }

    pub fn backward(&mut self, dvalue: f64, y_true: Vec<f64>) {
        self.dinputs = y_true.iter().map(|x| -x * dvalue).collect();
    }
}

pub trait Activation {
    fn new() -> Self;
    fn forward(&mut self, inputs: &Vec<f64>);
    fn backward(&mut self, dvalues: &Vec<f64>);
}

pub struct ActivationReLu {
    pub output: Vec<f64>,
    pub dinputs: Vec<f64>,
}

impl Activation for ActivationReLu {
    fn new() -> Self {
        Self {
            output: vec![],
            dinputs: vec![],
        }
    }
    fn forward(&mut self, inputs: &Vec<f64>) {
        self.output = inputs.iter().map(|x| x.max(0.0)).collect();
    }
    fn backward(&mut self, dvalues: &Vec<f64>) {
        self.dinputs = self.output.iter().zip(dvalues.iter()).map(|(output, dvalue)| if output > &0.0 { output * dvalue} else { 0.0 }).collect();
    }
}

pub struct ActivationSoftMax {
    pub output: Vec<f64>,
}

impl Activation for ActivationSoftMax {
    fn new() -> Self {
        Self {
            output: vec![],
        }
    }

    fn forward(&mut self, inputs: &Vec<f64>) {
        let max = inputs.iter().copied().reduce(f64::max).unwrap();
        let tiny_vals: Vec<f64> = inputs.iter().map(|x| x-max).collect();
        let exp_outputs: Vec<f64> = tiny_vals.iter().map(|x| x.exp()).collect();
        let sum: f64 = exp_outputs.iter().sum();
        let norm_values: Vec<f64> = exp_outputs.iter().map(|x| x/sum).collect();
        self.output = norm_values;
    }

    fn backward(&mut self, dvalues: &Vec<f64>) {
        
    }
}

pub enum Activations {
    ReLu(ActivationReLu),
    SoftMax(ActivationSoftMax),
}

impl Activations {
    pub fn forward(&mut self, inputs: &Vec<f64>) {
        match self {
            Activations::ReLu(relu) => relu.forward(inputs),
            Activations::SoftMax(softmax) => softmax.forward(inputs),
        }
    }

    pub fn output(&self) -> Vec<f64> {
        match self {
            Activations::ReLu(relu) => relu.output.clone(),
            Activations::SoftMax(softmax) => softmax.output.clone(),
        }
    }
}
pub fn layer_outputs(inputs: &Vec<f64>, weights: &Vec<Vec<f64>>, biases: &Vec<f64>) -> Vec<f64> {
    dot_matrix_vector(&inputs, &weights).iter().zip(biases.iter()).map(|(a, b)| a + b).collect()
}

// fn batch_feedforward(inputs: &Vec<Vec<f64>>, weights: &Vec<Vec<f64>>, biases: &Vec<f64>) -> Vec<Vec<f64>> {
//     matrix_product(inputs, weights).iter().map(|x| x.iter().zip(biases.iter()).map(|(a, b)| a + b).collect()).collect()
// }

pub fn dot(a_vec: &Vec<f64>, b_vec: &Vec<f64>) -> f64 {
    let mut output = 0.0;
    for (a, b) in a_vec.iter().zip(b_vec.iter()) {
        output += a * b;
    }
    output
}

pub fn dot_matrix_vector(inputs: &Vec<f64>, layer_weights: &Vec<Vec<f64>>) -> Vec<f64> {
    let mut outputs = vec![];
    for weights in layer_weights {
        let output = dot(inputs, weights);
        outputs.push(output);
    }
    outputs
}

// fn matrix_product(batch_inputs: &Vec<Vec<f64>>, weights: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
//     let mut outputs = vec![];
//     for inputs in batch_inputs {
//         let output = dot_matrix_vector(inputs, weights);
//         outputs.push(output);
//     }
//     outputs
// }