import numpy as np
from PIL import Image
import sys
import os

def convert_bin_to_image(sample_idx=0):
    """
    Convert binary output file to image
    
    Args:
        sample_idx: Index of sample to convert (default: 0)
    """
    
    bin_file_path = "./data/outputs.bin"
    
    if not os.path.exists(bin_file_path):
        print(f"Error: File {bin_file_path} not found")
        return
    
    # Calculate expected file size for one sample
    # 512 * 512 * 3 channels * 4 bytes per float
    expected_size_per_sample = 512 * 512 * 3 * 4
    
    # Check file size
    file_size = os.path.getsize(bin_file_path)
    num_samples = file_size // expected_size_per_sample
    
    print(f"File size: {file_size} bytes")
    print(f"Expected size per sample: {expected_size_per_sample} bytes")
    print(f"Number of samples detected: {num_samples}")
    
    if file_size % expected_size_per_sample != 0:
        print("Warning: File size is not a multiple of expected sample size")
    
    if sample_idx >= num_samples:
        print(f"Error: Sample index {sample_idx} is out of range (0-{num_samples-1})")
        return
    
    with open(bin_file_path, 'rb') as f:
        # Skip to the desired sample
        f.seek(sample_idx * expected_size_per_sample)
        
        # Read one sample (512*512*3 floats)
        data = f.read(expected_size_per_sample)
        
        if len(data) != expected_size_per_sample:
            print(f"Error: Could not read full sample. Read {len(data)} bytes, expected {expected_size_per_sample}")
            return
        
        # Convert binary data to numpy array
        img_array = np.frombuffer(data, dtype=np.float32).reshape((3, 512, 512))
        
        # Convert from CHW to HWC format (Height, Width, Channels)
        img_array = np.transpose(img_array, (1, 2, 0))
        
        # Normalize to 0-255 range
        img_min = img_array.min()
        img_max = img_array.max()
        print(f"Data range: [{img_min:.6f}, {img_max:.6f}]")
        
        if img_max > img_min:
            img_normalized = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_normalized = np.full_like(img_array, 128, dtype=np.uint8)
        
        # Create PIL Image
        img = Image.fromarray(img_normalized, 'RGB')
        
        # Generate output filename
        output_image_path = f"images/output_{sample_idx:04d}.png"
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_image_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save image
        img.save(output_image_path)
        print(f"Image saved to: {output_image_path}")
        
        return img_array

def convert_all_samples():
    """Convert all samples in the binary file to separate images"""
    
    bin_file_path = "./data/outputs.bin"
    
    if not os.path.exists(bin_file_path):
        print(f"Error: File {bin_file_path} not found")
        return
    
    # Calculate number of samples
    file_size = os.path.getsize(bin_file_path)
    expected_size_per_sample = 512 * 512 * 3 * 4
    num_samples = file_size // expected_size_per_sample
    
    print(f"Converting {num_samples} samples...")
    
    for i in range(num_samples):
        print(f"Converting sample {i+1}/{num_samples}...")
        convert_bin_to_image(i)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        convert_all_samples()
    else:
        sample_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        convert_bin_to_image(sample_idx)
        
    print("Usage:")
    print("  python bin2img.py          # Convert sample 0")
    print("  python bin2img.py 5        # Convert sample 5") 
    print("  python bin2img.py --all    # Convert all samples")
