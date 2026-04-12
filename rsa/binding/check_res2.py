import sys
import nibabel as nib
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else input("Path to .nii.gz: ").strip()

img      = nib.load(path)
zooms    = img.header.get_zooms()
vox_size = np.abs(np.diag(img.affine)[:3])

print(f"\nFile   : {path}")
print(f"Shape  : {img.shape}")
print(f"Zooms  : {zooms}")
print(f"Affine voxel sizes: {vox_size}")

is_res2 = np.allclose(vox_size, 2.0, atol=0.1)
print(f"\nres-2  : {'YES' if is_res2 else 'NO'} (voxel sizes {vox_size[0]:.2f} x {vox_size[1]:.2f} x {vox_size[2]:.2f} mm)")