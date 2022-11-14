use rand::{thread_rng, distributions::Uniform, Rng};

pub fn circle_dataset(r: f64, num_samples: u64) -> Vec<(Vec<f64>, usize)> {
    let mut rng = thread_rng();
    let range = Uniform::new(r * -2.0, r * 2.0);

    let random_samples: Vec<Vec<f64>> = (0..num_samples).map(|_| vec![rng.sample(range), rng.sample(range)]).collect();

    let output_samples: Vec<(Vec<f64>, usize)> = random_samples.iter().map(|vec| (vec.to_owned(), (vec[0].powf(2.0) + vec[1].powf(2.0) <= r.powf(2.0)) as usize)).collect();
    output_samples
}