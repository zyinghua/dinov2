import os
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm

def gaussian_noise(x, std, rng):
    c = std/255
    x = np.array(x) / 255.
    noisy_image = np.clip(x + rng.normal(size=x.shape, scale=c), 0, 1) * 255
    return noisy_image.astype(np.uint8)

def add_noise_to_dataset(source_root, target_root, noise):
    seed = 42
    rng = np.random.default_rng(seed)
    # Copy the directory structure
    if os.path.exists(target_root):
        shutil.rmtree(target_root)
    shutil.copytree(source_root, target_root, ignore=shutil.ignore_patterns('*.*'))
    
    # Copy all metadata files (.npy and .txt files) from the root directory
    metadata_files = [
        'class-ids-TRAIN.npy',
        'class-ids-VAL.npy',
        'class-names-TRAIN.npy',
        'class-names-VAL.npy',
        'entries-TEST.npy',
        'entries-TRAIN.npy',
        'entries-VAL.npy',
        'labels.txt'
    ]
    for metadata_file in metadata_files:
        source_file = os.path.join(source_root, metadata_file)
        target_file = os.path.join(target_root, metadata_file)
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            print(f"Copied {metadata_file}")
    
    noise_type, noise_param = noise.split("_")
    noise_param = float(noise_param)
    if noise_type == "gauss":
        add_noise = gaussian_noise
    else:
        raise ValueError(f"Noise type {noise_type} not supported")
    # Loop through each set (train, test, val)
    for set_name in ['train', 'val']:
        set_path = os.path.join(source_root, set_name)
        # Loop through each class folder
        for class_folder in tqdm(os.listdir(set_path)):
            class_path = os.path.join(set_path, class_folder)
            target_class_path = os.path.join(target_root, set_name, class_folder)
            # Loop through each image
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                base_name = os.path.splitext(image_file)[0]
                target_image_path = os.path.join(target_class_path, base_name + ".png")
                
                # Open image, add noise, and save it
                with Image.open(image_path).convert("RGB") as img:
                    noisy_image = add_noise(img, noise_param, rng)
                    Image.fromarray(noisy_image).save(target_image_path)


add_noise_to_dataset('/root/autodl-tmp/mini-imagenet', "/root/autodl-tmp/noisy_mini-imagenet-gauss25", "gauss_25")
add_noise_to_dataset('/root/autodl-tmp/mini-imagenet', "/root/autodl-tmp/noisy_mini-imagenet-gauss50", "gauss_50")
add_noise_to_dataset('/root/autodl-tmp/mini-imagenet', "/root/autodl-tmp/noisy_mini-imagenet-gauss75", "gauss_75")
add_noise_to_dataset('/root/autodl-tmp/mini-imagenet', "/root/autodl-tmp/noisy_mini-imagenet-gauss100", "gauss_100")