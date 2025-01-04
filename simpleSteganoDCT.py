"""
SimpleSteganoDCT
Sign-based DCT steganography with Reed–Solomon error correction
This script provides a straightforward CLI for embedding and extracting text
in PNG images using mid-frequency DCT coefficients in the luminance channel.
It also calculates basic quality metrics (PSNR, SSIM) to assess the impact
on the image.
Author: Mounir IDRASSI <mounir@idrix.fr>
License: Apache License 2.0
Date: 2025-01-04
"""

import json
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from reedsolo import RSCodec
import argparse
import os

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
                "embedding_strength": 3.0,  # Alpha value for sign-based embedding
                "redundancy": 4,  # Number of coefficients per block for embedding
                "rs_error_correction": 20
            }

        self.block_size = self.config.get("block_size", 8)
        self.sync_marker = self.config.get("sync_marker", "10101010")
        self.embedding_strength = self.config.get("embedding_strength", 3.0)
        self.redundancy = self.config.get("redundancy", 4)
        self.rs_error_correction = self.config.get("rs_error_correction", 30)
        self.rs_codec = RSCodec(self.rs_error_correction)

        # Define coefficient positions for embedding (mid-frequency coefficients)
        self.coeff_positions = [(2,2), (2,3), (3,2), (3,3)][:self.redundancy]        

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
        Embed a single bit using sign-based approach.
        """
        i, j = position
        c = dct_block[i, j]

        if bit == 1:
            # Force a positive sign
            if c <= 0:
                dct_block[i, j] = abs(c) + self.embedding_strength
        else:
            # Force a negative sign
            if c >= 0:
                dct_block[i, j] = -abs(c) - self.embedding_strength

        return dct_block

    def _embed_in_dct(self, y_channel, message):
        """
        Embed the entire message in the DCT coefficients using sign-based embedding.
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
        Extract a single bit using sign-based approach.
        """
        i, j = position
        c = dct_block[i, j]
        return 1 if c > 0 else 0

    def _extract_from_dct(self, y_channel, total_bits):
        """
        Extract total_bits bits from the DCT coefficients using sign-based extraction.
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
        self._validate_png_format(stego_image_path)
        try:
            # We know encoded2 is encoding of 4 bytes so we use get_rs_encoded_length 
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
    parser = argparse.ArgumentParser(description='Simple steganography for text embedding and extraction')
    subparsers = parser.add_subparsers(dest='mode', help='Operating mode')

    # Embed mode parser
    embed_parser = subparsers.add_parser('embed', help='Embed text in image')
    embed_parser.add_argument('input_image', help='Path to input image')
    embed_parser.add_argument('output_png_image', help='Path to output PNG image')
    embed_parser.add_argument('--text', '-t', default="©2025 SimpleStegano",
                           help='Text to embed (default: "©2025 SimpleStegano")')

    # Extract mode parser
    extract_parser = subparsers.add_parser('extract', help='Extract text from image')
    extract_parser.add_argument('input_png_image', help='Path to PNG image containing hidden text')

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
            # Validate input is PNG
            if not args.input_png_image.lower().endswith('.png'):
                raise ValueError("Input file must be a PNG image")

            # Validate input image exists and can be opened
            if not os.path.exists(args.input_png_image):
                raise ValueError(f"Input PNG image not found: {args.input_png_image}")

            message = steg.extract_text_steganography(args.input_png_image)
            print(f"Extracted message: {message}")

    except Exception as e:
        print(f"Error: {e}")

