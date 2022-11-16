mod dataset;
mod data_structures;

use dataset::circle_dataset;
use data_structures::*;

fn main() {
    let dataset = circle_dataset(2.0, 10000);
    // let inputs = &dataset[0];

    // let mut dense1 = LayerDense::new(2, 5);
    // let mut activation1 = ActivationReLu::new();
    // let mut dense2 = LayerDense::new(5, 2);
    // let mut activation2 = ActivationSoftMax::new();
    // // let loss = Loss {};
    
    // dense1.forward(&inputs.0);
    // activation1.forward(&dense1.output);
    // dense2.forward(&activation1.output);
    // activation2.forward(&dense2.output);
    // let loss = Loss::forward(&activation2.output, inputs.1);
    // let accuracy = Accuracy::calculate(&activation2.output, inputs.1);
    
    // println!("{:?}", loss);
    // println!("{:?}", accuracy);
    let (inputs, answers): (Vec<Vec<f64>>, Vec<usize>) = dataset.iter().cloned().map(|(input, answer)| (input, answer)).unzip();

    let mut nn = NeuralNetwork { layers: vec![
        Layer {
            neurons: LayerDense::new(2, 30),
            activation: Activations::ReLu(ActivationReLu::new())
        },
        Layer {
            neurons: LayerDense::new(30, 30),
            activation: Activations::ReLu(ActivationReLu::new())
        },
        Layer {
            neurons: LayerDense::new(30, 2),
            activation: Activations::SoftMax(ActivationSoftMax::new())
        }
    ], loss: 999999999.9 };
    nn.train(&inputs, &answers, 1, 0.05);

    let test_data = circle_dataset(2.0, 50);
    let (inputs, answers): (Vec<Vec<f64>>, Vec<usize>) = test_data.iter().cloned().map(|(input, answer)| (input, answer)).unzip();
    
    nn.evaluate(&inputs, &answers);
}