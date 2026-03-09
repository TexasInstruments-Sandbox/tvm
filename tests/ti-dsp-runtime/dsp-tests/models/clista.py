import torch
import torch.nn as nn
import numpy as np

class CLISTA_DoA(nn.Module):
    """
    Convolutional LISTA for Direction of Arrival Estimation
    Designed for TI AWR6844 with 16 virtual antennas
    
    Architecture:
    - Input: [Batch, 2, 16] where 2 = I/Q channels, 16 = virtual antennas
    - Output: [Batch, num_atoms*2, 1] sparse angular spectrum (complex-valued)
    
    The model learns to solve:
        min_x ||x||_1 subject to ||Dx - y||_2^2 < epsilon
    where D is the learned dictionary (steering vectors)
    """
    
    def __init__(self,
                 num_iterations=8,
                 filter_size=9,  # noqa: ARG002  # kept for API compatibility
                 num_atoms=64,
                 num_antennas=16,
                 step_size=0.1):
        """
        Args:
            num_iterations: Number of ISTA unrolling iterations
            filter_size: Convolution kernel size (should be odd)
            num_atoms: Number of dictionary atoms (angular bins)
            num_antennas: Number of virtual antennas (16 for AWR6844)
            step_size: Initial gradient descent step size
        """
        super(CLISTA_DoA, self).__init__()
        
        self.num_iterations = num_iterations
        self.num_atoms = num_atoms
        self.num_antennas = num_antennas
        
        # Encoder: Maps antenna measurements to sparse angular domain
        # Input: [B, 2, 16] -> Output: [B, num_atoms*2, 1]
        self.encode = nn.Conv1d(
            in_channels=2,
            out_channels=num_atoms * 2,  # *2 for complex (real/imag pairs)
            kernel_size=num_antennas,     # Full receptive field
            stride=1,
            padding=0,
            bias=False
        )
        
        # Decoder: Reconstructs antenna measurements from sparse code
        # This represents the learned dictionary D
        # Input: [B, num_atoms*2, 1] -> Output: [B, 2, 16]
        self.decode = nn.ConvTranspose1d(
            in_channels=num_atoms * 2,
            out_channels=2,
            kernel_size=num_antennas,
            stride=1,
            padding=0,
            bias=False
        )
        
        # Learnable step size for gradient descent
        self.step_size = nn.Parameter(torch.tensor(step_size))
        
        # Learnable thresholds for each iteration (allows adaptive sparsity)
        self.thresholds = nn.Parameter(torch.ones(num_iterations) * 0.05)
        
        # Initialize weights with reasonable values
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize encoder/decoder with small random values"""
        nn.init.xavier_uniform_(self.encode.weight, gain=0.1)
        nn.init.xavier_uniform_(self.decode.weight, gain=0.1)
    
    def soft_threshold_complex(self, x, threshold):
        """
        Complex-valued soft thresholding (shrinkage operator)
        
        For each complex number z = a + bi:
        - Compute magnitude: |z| = sqrt(a^2 + b^2)
        - Shrink: |z|_new = max(|z| - threshold, 0)
        - Preserve phase: z_new = z * (|z|_new / |z|)
        
        Args:
            x: [Batch, num_atoms*2, Spatial] where channels are (real, imag) pairs
            threshold: scalar threshold value
            
        Returns:
            Thresholded tensor with same shape as x
        """
        batch, channels, spatial = x.shape
        
        # Reshape to separate real/imag pairs: [B, num_atoms, 2, Spatial]
        x_complex = x.view(batch, channels // 2, 2, spatial)
        
        # Extract real and imaginary parts
        real = x_complex[:, :, 0, :]  # [B, num_atoms, Spatial]
        imag = x_complex[:, :, 1, :]  # [B, num_atoms, Spatial]
        
        # Compute magnitude: sqrt(real^2 + imag^2)
        magnitude = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        
        # Soft thresholding: max(magnitude - threshold, 0)
        magnitude_shrunk = torch.relu(magnitude - threshold)
        
        # Compute scaling factor (avoid division by zero)
        scale = magnitude_shrunk / (magnitude + 1e-8)
        
        # Apply scaling to both real and imaginary parts
        real_shrunk = real * scale
        imag_shrunk = imag * scale
        
        # Recombine into [B, num_atoms, 2, Spatial]
        x_complex_shrunk = torch.stack([real_shrunk, imag_shrunk], dim=2)
        
        # Reshape back to [B, num_atoms*2, Spatial]
        return x_complex_shrunk.view(batch, channels, spatial)
    
    def forward(self, y):
        """
        Forward pass: Unrolled ISTA iterations
        
        Args:
            y: Input radar signal [Batch, 2, 16]
               - Channel 0: I (In-phase)
               - Channel 1: Q (Quadrature)
               - Spatial dim: 16 virtual antennas
        
        Returns:
            x: Sparse angular spectrum [Batch, num_atoms*2, 1]
               - Channels are (real, imag) pairs
               - Peaks indicate DoA angles
        """
        # Initial sparse code estimate: x_0 = Threshold(W_e^T * y)
        x = self.encode(y)  # [B, num_atoms*2, 1]
        x = self.soft_threshold_complex(x, self.thresholds[0])
        
        # Unrolled ISTA iterations
        for iteration in range(1, self.num_iterations):
            # Reconstruction: y_hat = D * x
            y_reconstructed = self.decode(x)  # [B, 2, 16]
            
            # Compute residual: r = y - y_hat
            residual = y - y_reconstructed
            
            # Gradient step: x = x + step_size * W_e^T * residual
            gradient = self.encode(residual)
            x = x + self.step_size * gradient
            
            # Proximal operator: soft thresholding
            x = self.soft_threshold_complex(x, self.thresholds[iteration])
        
        return x
    
    def get_angular_spectrum(self, x):
        """
        Convert complex sparse code to real-valued angular spectrum (magnitude)
        
        Args:
            x: Sparse code [Batch, num_atoms*2, 1]
        
        Returns:
            spectrum: [Batch, num_atoms] - magnitude at each angle bin
        """
        batch, channels, spatial = x.shape
        x_complex = x.view(batch, channels // 2, 2, spatial)
        
        # Compute magnitude
        real = x_complex[:, :, 0, :]
        imag = x_complex[:, :, 1, :]
        magnitude = torch.sqrt(real ** 2 + imag ** 2)
        
        return magnitude.squeeze(-1)  # [B, num_atoms]
    
    def estimate_doa(self, x, angle_range=(-60, 60)):
        """
        Extract DoA angles from sparse spectrum
        
        Args:
            x: Sparse code [Batch, num_atoms*2, 1]
            angle_range: Tuple of (min_angle, max_angle) in degrees
        
        Returns:
            angles: List of detected angles for each batch
            amplitudes: Corresponding amplitudes
        """
        spectrum = self.get_angular_spectrum(x)  # [B, num_atoms]
        
        # Convert atom indices to angles
        angles_grid = torch.linspace(angle_range[0], angle_range[1], 
                                     self.num_atoms, device=x.device)
        
        batch_size = spectrum.shape[0]
        detected_angles = []
        detected_amplitudes = []
        
        for b in range(batch_size):
            # Find peaks above threshold (simple peak detection)
            spec = spectrum[b]
            threshold = 0.3 * spec.max()  # 30% of max
            peaks = (spec > threshold).nonzero(as_tuple=True)[0]
            
            if len(peaks) > 0:
                angles = angles_grid[peaks].cpu().numpy()
                amps = spec[peaks].cpu().numpy()
                detected_angles.append(angles)
                detected_amplitudes.append(amps)
            else:
                detected_angles.append(np.array([]))
                detected_amplitudes.append(np.array([]))
        
        return detected_angles, detected_amplitudes


def create_test_input(batch_size=1, num_antennas=16, num_targets=2, snr_db=10):
    """
    Generate synthetic radar data for testing
    
    Args:
        batch_size: Number of samples
        num_antennas: Number of virtual antennas
        num_targets: Number of targets to simulate
        snr_db: Signal-to-noise ratio in dB
    
    Returns:
        y: Simulated radar signal [batch_size, 2, num_antennas]
        true_angles: Ground truth angles in degrees
    """
    # Random target angles between -60 and 60 degrees
    true_angles = (torch.rand(batch_size, num_targets) - 0.5) * 120  # [-60, 60]
    
    # Random target amplitudes
    amplitudes = torch.rand(batch_size, num_targets) * 0.5 + 0.5  # [0.5, 1.0]
    
    # Antenna array geometry (uniform linear array)
    antenna_positions = torch.arange(num_antennas, dtype=torch.float32)
    wavelength = 1.0  # Normalized
    d = wavelength / 2  # Half-wavelength spacing
    
    # Initialize signal
    y_complex = torch.zeros(batch_size, num_antennas, dtype=torch.complex64)
    
    for b in range(batch_size):
        for t in range(num_targets):
            angle_rad = true_angles[b, t] * np.pi / 180
            
            # Steering vector: exp(j * 2π * d * n * sin(θ) / λ)
            phase = 2 * np.pi * d * antenna_positions * torch.sin(torch.tensor(angle_rad)) / wavelength
            steering_vector = torch.exp(1j * phase)
            
            # Add target contribution
            y_complex[b] += amplitudes[b, t] * steering_vector
    
    # Add noise
    noise_power = 10 ** (-snr_db / 10)
    noise = torch.randn(batch_size, num_antennas, dtype=torch.complex64) * np.sqrt(noise_power / 2)
    y_complex += noise
    
    # Convert to I/Q format: [batch, 2, num_antennas]
    y = torch.stack([y_complex.real, y_complex.imag], dim=1)
    
    return y, true_angles


def main():
    """
    Test the model with random weights
    """
    print("=" * 70)
    print("CLISTA-DoA Model for TI AWR6844")
    print("=" * 70)
    
    # Model configuration
    config = {
        'num_iterations': 10,
        'filter_size': 9,
        'num_atoms': 64,
        'num_antennas': 16,
        'step_size': 0.1
    }
    
    print("\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create model
    model = CLISTA_DoA(**config)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size (FP32): {total_params * 4 / 1024:.2f} KB")
    print(f"  Model size (INT8): {total_params / 1024:.2f} KB (estimated)")
    
    # Generate test input
    print("\n" + "=" * 70)
    print("Testing with Synthetic Data")
    print("=" * 70)
    
    batch_size = 4
    y_test, true_angles = create_test_input(
        batch_size=batch_size,
        num_antennas=16,
        num_targets=2,
        snr_db=10
    )
    
    print(f"\nInput shape: {y_test.shape}")
    print("True angles (degrees):")
    for b in range(batch_size):
        print(f"  Sample {b}: {true_angles[b].numpy()}")
    
    # Run inference
    with torch.no_grad():
        output = model(y_test)
    
    print(f"\nOutput shape: {output.shape}")
    
    # Get angular spectrum
    spectrum = model.get_angular_spectrum(output)
    print(f"Angular spectrum shape: {spectrum.shape}")
    
    # Estimate DoA
    detected_angles, detected_amps = model.estimate_doa(output)
    
    print("\nDetected Angles:")
    for b in range(batch_size):
        print(f"  Sample {b}:")
        print(f"    True: {true_angles[b].numpy()}")
        print(f"    Detected: {detected_angles[b]}")
        print(f"    Amplitudes: {detected_amps[b]}")
    
    # Export to ONNX for TVM
    print("\n" + "=" * 70)
    print("Exporting to ONNX")
    print("=" * 70)
    
    onnx_path = "clista_doa.onnx"
    dummy_input = torch.randn(1, 2, 16)

    torch.onnx.export(
        model,
        (dummy_input,),  # args must be tuple
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['radar_signal'],
        output_names=['sparse_spectrum'],
        dynamic_axes={
            'radar_signal': {0: 'batch_size'},
            'sparse_spectrum': {0: 'batch_size'}
        }
    )
    
    print(f"\nModel exported to: {onnx_path}")
    print("Input: 'radar_signal' [batch, 2, 16]")
    print(f"Output: 'sparse_spectrum' [batch, {config['num_atoms']*2}, 1]")
    
    # Test with TorchScript (alternative to ONNX)
    print("\n" + "=" * 70)
    print("Creating TorchScript")
    print("=" * 70)
    
    scripted_model: torch.jit.ScriptModule = torch.jit.trace(model, dummy_input)  # pyright: ignore[reportAssignmentType]
    torchscript_path = "clista_doa.pt"
    scripted_model.save(torchscript_path)
    
    print(f"\nTorchScript model saved to: {torchscript_path}")
    
    # Verify TorchScript model
    loaded_model = torch.jit.load(torchscript_path)
    with torch.no_grad():
        output_original = model(dummy_input)
        output_scripted = loaded_model(dummy_input)
        max_diff = torch.max(torch.abs(output_original - output_scripted)).item()
    
    print(f"TorchScript verification - Max difference: {max_diff:.2e}")
    
    print("\n" + "=" * 70)
    print("Ready for TVM Relax Compilation!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Load ONNX model in TVM:")
    print("   import tvm.relay as relay")
    print("   mod, params = relay.frontend.from_onnx(onnx.load('clista_doa.onnx'))")
    print("\n2. Compile for target (e.g., C66x DSP):")
    print("   target = tvm.target.Target('c66x')")
    print("   with tvm.transform.PassContext(opt_level=3):")
    print("       lib = relay.build(mod, target=target, params=params)")
    print("\n3. Deploy to AWR6844")
    

if __name__ == "__main__":
    main()
