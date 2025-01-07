"""
SimpleSteganoDCT
QIM-based DCT steganography with JPEG compression resistance and Reed-Solomon error correction

This script provides a robust steganographic system for embedding and extracting text
in images using Quantization Index Modulation (QIM) in the DCT domain. The implementation
is specifically designed to survive JPEG compression by aligning the embedding process
with standard JPEG quantization matrices.

Key features:
- QIM-based embedding in mid-frequency DCT coefficients
- JPEG compression resistance (optimized for ~70% quality)
- Reed-Solomon error correction for improved reliability
- Synchronization markers for accurate message recovery
- Quality metrics calculation (PSNR, SSIM)

The system uses the luminance channel for embedding and implements a two-step payload
structure with length encoding and error correction to ensure reliable message recovery
even after compression.

Usage:
  embed:   python simpleSteganoDCT.py embed input_image output_png_image --text "message"
  extract: python simpleSteganoDCT.py extract input_image

Author: Mounir IDRASSI <mounir@idrix.fr>
License: Apache License 2.0
Date: 2025-01-07
"""

import json
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from reedsolo import RSCodec
import argparse
import os

#############################################
#  JPEG Luminance Quantization for Q=50
#############################################
LUMINANCE_QUANT_MATRIX_50 = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99]
], dtype=np.float32)

#############################################
#  Scale factor for Q=70
#  Formula for Q >= 50 is scale = (200 - 2*Q) / 100
#  For Q=70, scale = (200 - 140) / 100 = 0.6
#############################################
SCALE_Q70 = 0.6

# Create a Q=70 matrix by scaling the Q=50 matrix
LUMINANCE_QUANT_MATRIX_70 = np.floor(LUMINANCE_QUANT_MATRIX_50 * SCALE_Q70 + 0.5).astype(np.float32)

class SimpleStegano:
    def __init__(self, config_path="config.json"):
        """
        Initialize the SimpleStegano system with configuration.
        """
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Default configuration if file not found
            self.config = {
                "block_size": 8,
                "sync_marker": "10101010",
                "redundancy": 4,  # Number of coefficients per block for embedding
                "rs_error_correction": 20,
                "coeff_positions": [[2,2], [2,3], [3,2], [3,3]],
                "alpha": 1.5
            }

        self.block_size = self.config.get("block_size", 8)
        self.sync_marker = self.config.get("sync_marker", "10101010")
        self.redundancy = self.config.get("redundancy", 4)
        self.rs_error_correction = self.config.get("rs_error_correction", 30)
        self.rs_codec = RSCodec(self.rs_error_correction)

        # Define default coefficient positions for embedding (mid-frequency coefficients)
        default_positions = [(2,2), (2,3), (3,2), (3,3)]
        positions_data = self.config.get("coeff_positions", default_positions)
        try:
            # Convert list of lists to list of tuples if needed since json doesn't support tuples
            positions = [tuple(pos) if isinstance(pos, list) else pos 
                        for pos in positions_data]
            self.coeff_positions = positions[:self.redundancy]
        except (TypeError, ValueError):
            self.coeff_positions = default_positions[:self.redundancy]

        # QIM scale factor alpha. Increase if you want even stronger embedding
        self.alpha = self.config.get("alpha", 1.5)

    # a method that returns the length of rs_codec encoding of 4 bytes
    def get_rs_encoded_length(self):
        return len(self.rs_codec.encode(b'abcd'))

    def _preprocess_image(self, image_path):
        """
        Preprocess the image: Convert to YCrCb color space and extract Y-channel.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Unable to read image from path: {image_path}")
            ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y_channel = ycrcb_image[:, :, 0]
            return y_channel, ycrcb_image
        except Exception as e:
            raise RuntimeError(f"Image preprocessing failed: {e}")

    def _embed_bit_in_coefficient(self, dct_block, position, bit):
        """
        Embed a single bit in dct_block at position using QIM with respect
        to the JPEG luminance quantization matrix scaled for ~70% quality.
        """
        i, j = position
        c = dct_block[i, j]

        # Get the quant step from the scaled matrix
        qstep_base = LUMINANCE_QUANT_MATRIX_70[i, j]
        qstep = self.alpha * qstep_base  # scale up for more robust embedding

        # Compute integer index
        k = int(round(c / qstep))

        # Force parity of k to match the bit
        # If bit == 1 -> k is odd, else even
        if bit == 1:
            if (k % 2) == 0:
                k += 1
        else:
            if (k % 2) == 1:
                k += 1

        # Reconstruct the modified coefficient
        c_prime = k * qstep
        dct_block[i, j] = c_prime
        return dct_block

    def _embed_in_dct(self, y_channel, message):
        """
        Embed the entire message in the DCT coefficients using QIM-based embedding.
        message should be a bytes-like object.
        """
        try:
            height, width = y_channel.shape
            # Convert the entire message to a stream of bits
            message_bits = ''.join(format(byte, '08b') for byte in message)
            message_index = 0
            message_length = len(message_bits)

            for i in range(0, height, self.block_size):
                for j in range(0, width, self.block_size):
                    if message_index >= message_length:
                        break

                    block = y_channel[i:i+self.block_size, j:j+self.block_size]
                    if block.shape == (self.block_size, self.block_size):
                        dct_block = cv2.dct(np.float32(block))

                        # Embed bits using multiple coefficients
                        for pos in self.coeff_positions:
                            if message_index < message_length:
                                bit = int(message_bits[message_index])
                                dct_block = self._embed_bit_in_coefficient(dct_block, pos, bit)
                                message_index += 1

                        # Apply IDCT and clamp values
                        modified_block = cv2.idct(dct_block)
                        modified_block = np.clip(modified_block, 0, 255)
                        y_channel[i:i+self.block_size, j:j+self.block_size] = modified_block

            return y_channel
        except Exception as e:
            raise RuntimeError(f"Message embedding failed: {e}")

    def _extract_bit_from_coefficient(self, dct_block, position):
        """
        Extract a single bit using QIM-based approach (checking parity of quantized index).
        """
        i, j = position
        c = dct_block[i, j]

        qstep_base = LUMINANCE_QUANT_MATRIX_70[i, j]
        qstep = self.alpha * qstep_base

        k = int(round(c / qstep))
        return 1 if (k % 2) == 1 else 0

    def _extract_from_dct(self, y_channel, total_bits):
        """
        Extract total_bits bits from the DCT coefficients using QIM-based extraction.
        """
        try:
            height, width = y_channel.shape
            extracted_bits = []

            for i in range(0, height, self.block_size):
                for j in range(0, width, self.block_size):
                    if len(extracted_bits) >= total_bits:
                        break

                    block = y_channel[i:i+self.block_size, j:j+self.block_size]
                    if block.shape == (self.block_size, self.block_size):
                        dct_block = cv2.dct(np.float32(block))

                        # Extract bits from multiple coefficients
                        for pos in self.coeff_positions:
                            if len(extracted_bits) < total_bits:
                                bit = self._extract_bit_from_coefficient(dct_block, pos)
                                extracted_bits.append(str(bit))

            return ''.join(extracted_bits)
        except Exception as e:
            raise RuntimeError(f"Message extraction failed: {e}")

    def embed_text_steganography(self, image_path, text_value, output_path):
        """
        Embed the text in the image using a two-step approach:
          1. Encode the length of (sync_marker + message) in 4 bytes, and apply RS encoding (encoded2).
          2. Then encode the actual message (encoded1).
          3. final_payload = encoded2 || encoded1
        """
        try:
            # 1) Prepare the actual message with sync marker
            message_with_marker = self.sync_marker + text_value
            encoded1 = self.rs_codec.encode(message_with_marker.encode("utf-8"))

            # 2) Prepare length bytes (4 bytes big-endian) and encode with RS
            length_bytes = len(encoded1).to_bytes(4, 'big')
            encoded2 = self.rs_codec.encode(length_bytes)

            # final payload to embed
            final_payload = encoded2 + encoded1

            # 3) Embed in the image
            y_channel, ycrcb_image = self._preprocess_image(image_path)
            stego_y_channel = self._embed_in_dct(y_channel, final_payload)

            # Reconstruct the image
            ycrcb_image[:, :, 0] = stego_y_channel
            stego_image = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2BGR)
            cv2.imwrite(output_path, stego_image)

            # Some quality metrics
            original_image = cv2.imread(image_path)
            psnr_value = psnr(original_image, stego_image)
            ssim_value = ssim(original_image, stego_image, channel_axis=2)
            print(f"PSNR: {psnr_value}, SSIM: {ssim_value}")

            return output_path
        except Exception as e:
            raise RuntimeError(f"Text embedding failed: {e}")
        
    def _validate_png_format(self, image_path):  
        """  
        Validate that the image is in PNG format.  
        """  
        if not image_path.lower().endswith('.png'):  
            raise ValueError("Image must be in PNG format")  

        # Additional PNG signature check  
        try:  
            with open(image_path, 'rb') as f:  
                png_signature = b'\x89PNG\r\n\x1a\n'  
                file_signature = f.read(len(png_signature))  
                if file_signature != png_signature:  
                    raise ValueError("Invalid PNG file signature")  
        except Exception as e:  
            raise ValueError(f"Error validating PNG format: {e}")

    def extract_text_steganography(self, stego_image_path):
        """
        Extract the two-part payload:
         - First get_rs_encoded_length bytes (encoded2) -> decode using RS -> yields 4 bytes (the length).
         - Then extract the next (that length) bytes (encoded1) -> decode using RS -> yields original message string.
         - Validate sync marker.
        """

        try:
            # We know encoded2 is encoding of 4 bytes, so we use get_rs_encoded_length 
            # to get the length in bytes. Multiply by 8 to get the length in bits.
            encoded2_size_in_bits = self.get_rs_encoded_length() * 8

            # Step 1: Extract the first bits for encoded2
            y_channel, _ = self._preprocess_image(stego_image_path)
            extracted_bits_for_encoded2 = self._extract_from_dct(y_channel, encoded2_size_in_bits)

            # Convert bits to bytes
            encoded2_bytes = [
                int(extracted_bits_for_encoded2[i:i+8], 2)
                for i in range(0, len(extracted_bits_for_encoded2), 8)
            ]
            decoded2 = self.rs_codec.decode(bytes(encoded2_bytes))[0]  # 4 raw bytes

            # This is the length of encoded1
            encoded1_length = int.from_bytes(decoded2, 'big')

            # Step 2: Extract the next encoded1_length bytes
            # We already extracted encoded2 bytes from the total stream. 
            # total payload bits so far is encoded2_size_in_bits. The next portion is:
            encoded1_size_in_bits = encoded1_length * 8
            extracted_bits_for_encoded1 = self._extract_from_dct(
                y_channel, 
                encoded2_size_in_bits + encoded1_size_in_bits  # extract up to this many bits in total
            )

            # We only need the bits after the first encoded2_size_in_bits
            extracted_bits_for_encoded1 = extracted_bits_for_encoded1[encoded2_size_in_bits:]

            # Convert bits to bytes
            encoded1_bytes = [
                int(extracted_bits_for_encoded1[i:i+8], 2)
                for i in range(0, len(extracted_bits_for_encoded1), 8)
            ]
            decoded1 = self.rs_codec.decode(bytes(encoded1_bytes))[0].decode("utf-8")

            # Validate sync marker
            if decoded1.startswith(self.sync_marker):
                return decoded1[len(self.sync_marker):]
            else:
                raise ValueError("Sync marker not found. Extraction failed.")
        except Exception as e:
            raise RuntimeError(f"Text extraction failed: {e}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='QIM-based steganography for text embedding and extraction')
    subparsers = parser.add_subparsers(dest='mode', help='Operating mode')

    # Embed mode parser
    embed_parser = subparsers.add_parser('embed', help='Embed text in image')
    embed_parser.add_argument('input_image', help='Path to input image')
    embed_parser.add_argument('output_png_image', help='Path to output PNG image')
    embed_parser.add_argument('--text', '-t', default="©2025 SimpleStegano",
                           help='Text to embed (default: "©2025 SimpleStegano")')

    # Extract mode parser
    extract_parser = subparsers.add_parser('extract', help='Extract text from image')
    extract_parser.add_argument('input_image', help='Path to image containing hidden text')

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        exit(1)

    try:
        steg = SimpleStegano()

        if args.mode == 'embed':
            # Validate output path has .png extension
            if not args.output_png_image.lower().endswith('.png'):
                raise ValueError("Output file must have .png extension")

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(args.output_png_image)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Validate input image can be opened
            if not os.path.exists(args.input_image):
                raise ValueError(f"Input image not found: {args.input_image}")

            stego_path = steg.embed_text_steganography(
                image_path=args.input_image,
                text_value=args.text,
                output_path=args.output_png_image
            )
            print(f"Stego image saved to: {stego_path}")

            # Verify embedding by extracting
            message = steg.extract_text_steganography(stego_path)
            print(f"Verification - Extracted message: {message}")

        elif args.mode == 'extract':

            # Validate input image exists and can be opened
            if not os.path.exists(args.input_image):
                raise ValueError(f"Input PNG image not found: {args.input_image}")

            message = steg.extract_text_steganography(args.input_image)
            print(f"Extracted message: {message}")

    except Exception as e:
        print(f"Error: {e}")
