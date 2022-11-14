mod dataset;
mod data_structures;

use dataset::circle_dataset;
use data_structures::*;

fn main() {
    let dataset = circle_dataset(2.0, 500);
    let inputs = &dataset[0];

    let mut dense1 = LayerDense::new(2, 5);
    let mut activation1 = ActivationReLu::new();
    let mut dense2 = LayerDense::new(5, 2);
    let mut activation2 = ActivationSoftMax::new();
    // let loss = Loss {};
    
    dense1.forward(&inputs.0);
    activation1.forward(&dense1.output);
    dense2.forward(&activation1.output);
    activation2.forward(&dense2.output);
    let loss = Loss::calculate(&activation2.output, inputs.1);
    let accuracy = Accuracy::calculate(&activation2.output, inputs.1);
    
    println!("{:?}", loss);
    println!("{:?}", accuracy);
}