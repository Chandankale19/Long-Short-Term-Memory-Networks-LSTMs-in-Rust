use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set the device to CUDA if available; otherwise, use CPU.
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    // Define the neural network configuration
    let vs = nn::VarStore::new(device);

    // Input size, hidden size, and number of layers for the LSTM
    let input_size = 10;
    let hidden_size = 20;
    let num_layers = 2;
    let batch_size = 16;
    let seq_len = 5;

    // Create the LSTM model
    let lstm = nn::lstm(&vs.root(), input_size, hidden_size, nn::LSTMConfig {
        num_layers: num_layers,
        dropout: 0.2, // Dropout rate
        bidirectional: false, // Set true for bidirectional LSTM
    });

    // Fully connected layer to map LSTM output to predictions
    let fc = nn::linear(&vs.root(), hidden_size, 1, Default::default());

    // Generate dummy input data (batch_size, seq_len, input_size)
    let input = Tensor::randn(&[seq_len, batch_size, input_size], (Kind::Float, device));
    let (hidden_state, cell_state) = lstm.zero_state(batch_size, &device);

    // Forward pass through LSTM
    let (output, _) = lstm.seq(&input, (hidden_state, cell_state));
    let predictions = output.apply(&fc);

    println!("LSTM output shape: {:?}", output.size());
    println!("Predictions shape: {:?}", predictions.size());

    // Define a loss function and optimizer
    let target = Tensor::randn(&[seq_len, batch_size, 1], (Kind::Float, device)); // Dummy target
    let loss_fn = nn::mse_loss();
    let loss = loss_fn(&predictions, &target, tch::Reduction::Mean);
    println!("Initial loss: {:?}", loss);

    let mut optimizer = nn::Adam::default().build(&vs, 1e-3)?;

    // Training loop
    for epoch in 1..=10 {
        optimizer.zero_grad();
        let (output, _) = lstm.seq(&input, lstm.zero_state(batch_size, &device));
        let predictions = output.apply(&fc);
        let loss = loss_fn(&predictions, &target, tch::Reduction::Mean);
        loss.backward();
        optimizer.step();

        println!("Epoch {}: Loss: {:?}", epoch, f64::from(loss));
    }

    Ok(())
}
