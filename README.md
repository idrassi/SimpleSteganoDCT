# SimpleSteganoDCT

A robust steganographic system for embedding and extracting text in images using Quantization Index Modulation (QIM) in the DCT domain. This implementation is specifically designed to survive JPEG compression by aligning the embedding process with standard JPEG quantization matrices.

---

## Features

- Text embedding in images using DCT coefficients.
- QIM-based embedding in mid-frequency DCT coefficients
- JPEG compression resistance (optimized for ~70% quality)
- Reed-Solomon error correction for improved reliability
- Synchronization markers for accurate message recovery
- Quality metrics calculation (PSNR, SSIM)
- Configurable parameters via JSON configuration file

---

## Requirements

- **Python** 3.7+
- **Dependencies**:
  - OpenCV (cv2)
  - NumPy
  - scikit-image
  - reedsolo

## Installation

```bash
# Clone the repository
git clone https://github.com/idrassi/SimpleSteganoDCT
cd SimpleSteganoDCT

# Install required dependencies
pip install numpy opencv-python scikit-image reedsolo
```

---

## Usage

### Embedding Text

```bash
python simplestegano.py embed input_image.jpg output_stego.png --text "Your secret message"
```

### Extracting Text

```bash
python simplestegano.py extract stego_image.png
```

---

## Configuration

Configure the system via `config.json`:

```json
{
    "block_size": 8,
    "sync_marker": "10101010",
    "redundancy": 4,
    "rs_error_correction": 20,
    "coeff_positions": [[2,2], [2,3], [3,2], [3,3]],
    "alpha": 1.5
}
```

### Parameters
- **block_size**: Size of DCT blocks (default: 8).
- **sync_marker**: Binary pattern for synchronization.
- **embedding_strength**: Coefficient modification strength.
- **redundancy**: Number of coefficients used per block.
- **rs_error_correction**: Reed-Solomon error correction strength.
- **coeff_positions**: DCT coefficient positions for embedding
- **alpha**: QIM scale factor for embedding strength

---

## How It Works

### Preprocessing
- Converts input image to YCrCb color space
- Uses luminance (Y) channel for embedding
- Divides image into 8x8 pixel blocks for DCT transformation
- Aligns embedding with JPEG quantization matrix (optimized for 70% quality)

### Embedding
1. Message Preparation:
   - Adds synchronization marker to the message
   - Applies Reed-Solomon error correction encoding
   - Creates two-part payload: length information and encoded message

2. DCT Domain Processing:
   - Transforms each 8x8 block to DCT domain
   - Uses Quantization Index Modulation (QIM) in mid-frequency coefficients
   - Embeds message bits by modifying coefficient quantization indices
   - Applies inverse DCT to return to spatial domain

3. Quality Preservation:
   - Maintains image quality through careful coefficient selection
   - Automatically calculates and reports PSNR and SSIM metrics
   - Saves output in lossless PNG format

### Extraction
1. Message Recovery:
   - Extracts length information from initial blocks
   - Uses length to determine message boundary
   - Processes DCT coefficients to recover embedded bits

2. Error Correction:
   - Applies Reed-Solomon decoding to correct potential errors
   - Validates synchronization marker
   - Recovers original message from corrected data

3. Verification:
   - Checks message integrity through sync marker
   - Handles potential extraction errors gracefully
   - Returns decoded message in original format

## Quality Assessment

The system provides **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index) metrics to evaluate the impact on image quality after embedding.

---

## Limitations

- Output images must be in PNG format to prevent compression losses
- Maximum message length depends on input image dimensions
- Optimized for JPEG compression around 70% quality

---

## License

This project is licensed under the **Apache License 2.0**.

---

## Author

**Mounir IDRASSI**  
[mounir@idrix.fr](mailto:mounir@idrix.fr)

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository.
2. Create your feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{SimpleSteganoDCT,
  author = {Idrassi, Mounir},
  title = {SimpleSteganoDCT: QIM-based DCT steganography with Reed-Solomon error correction},
  year = {2025},
  url = {https://github.com/idrassi/SimpleSteganoDCT}
}
```