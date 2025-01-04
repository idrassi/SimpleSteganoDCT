# SimpleSteganoDCT

A Python implementation of sign-based DCT steganography with Reed-Solomon error correction for embedding and extracting text in PNG images.

---

## Features

- **Text embedding** in PNG images using DCT coefficients.
- **Sign-based steganography** in mid-frequency DCT coefficients.
- **Reed-Solomon error correction** for robust message recovery.
- **Sync marker validation** for reliable extraction.
- **Quality metrics calculation**: PSNR and SSIM.
- **Command-line interface** for easy usage.

---

## Requirements

- **Python** 3.7+
- **Dependencies**:
  - OpenCV (cv2)
  - NumPy
  - scikit-image
  - reedsolo

Install dependencies using:

```bash
pip install opencv-python numpy scikit-image reedsolo
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
    "embedding_strength": 3.0,
    "redundancy": 4,
    "rs_error_correction": 20
}
```

### Parameters
- **block_size**: Size of DCT blocks (default: 8).
- **sync_marker**: Binary pattern for synchronization.
- **embedding_strength**: Coefficient modification strength.
- **redundancy**: Number of coefficients used per block.
- **rs_error_correction**: Reed-Solomon error correction strength.

---

## How It Works

### Preprocessing
- Converts image to YCrCb color space and works with the Y-channel.

### Embedding
1. Applies DCT to 8x8 blocks.
2. Embeds message bits in mid-frequency coefficients using a sign-based approach.
3. Uses Reed-Solomon coding for error correction.
4. Includes length information and sync marker.

### Extraction
1. Extracts encoded length information.
2. Recovers message using Reed-Solomon decoding.
3. Validates sync marker.
4. Returns the original message.

---

## Quality Assessment

The system provides **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index) metrics to evaluate the impact on image quality after embedding.

---

## Limitations

- Only supports **PNG** output format.
- Message length is limited by image size.
- Works best with natural images.
- Requires sufficient image quality for reliable extraction.

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
  title = {SimpleSteganoDCT: Sign-based DCT steganography with Reed-Solomon error correction},
  year = {2025},
  url = {https://github.com/idrassi/SimpleSteganoDCT}
}
```