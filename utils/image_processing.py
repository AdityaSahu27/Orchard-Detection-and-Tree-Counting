import os
from PIL import Image
import rasterio
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import cv2


def normalize_band(band, hist_min, hist_max):
    band = np.clip(band, hist_min, hist_max)
    return ((band - hist_min) / (hist_max - hist_min) * 255).astype(np.uint8)

def process_image_patches(image_path, patch_folder):
    # Assuming that the .tiff image is split into smaller patches
    # img = Image.open(image_path)
    # patches = []  # List of patch image paths
    
    # # Dummy patch creation logic
    # patch_size = 256
    # for i in range(0, img.size[0], patch_size):
    #     for j in range(0, img.size[1], patch_size):
    #         box = (i, j, i + patch_size, j + patch_size)
    #         patch = img.crop(box)
    #         patch_path = os.path.join(patch_folder, f"patch_{i}_{j}.png")
    #         patch.save(patch_path)
    #         patches.append(patch_path)
    dataset = rasterio.open(image_path)
    image = dataset.read()
    Patches_path = []
    # Read histogram information from the XML file
    hist_min = [0, 0, 0, 0]
    hist_max = [255, 255, 255, 255]
    # Normalize each band using histogram information
    for i in range(image.shape[0]):
        image[i, :, :] = normalize_band(image[i, :, :], hist_min[i], hist_max[i])
    # Transpose the image to (height, width, bands) format
    large_image = np.transpose(image, (1, 2, 0))
    # image = np.transpose(image, (1, 2, 0))
    # Patchify the image
    patch_size = (256, 256, 3)  # Patch size
    patches = patchify(large_image, patch_size, step=256)
    # patches = patchify(image, patch_size, step=256)
    # Save the patches
    patch_index = 0
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j, 0, :, :, :]
            # Convert patch to uint8
            patch_uint8 = patch.astype(np.uint8)
            plt.imsave(os.path.join(patch_folder, f"patch_{patch_index}.png"), patch_uint8)
            Patches_path.append(os.path.join(patch_folder, f"patch_{patch_index}.png"))
            patch_index += 1
    print(f"Saved {patch_index} patches to {patch_folder}")
    return Patches_path

def apply_segmentation(patch_folder, segmented_folder):
    # Dummy instance segmentation for orchard boundaries
    # patch_files = os.listdir(patch_folder)
    tag = 0
    segmented_patches = []
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="EREBFGvIAB2M51sxA6gH"
    )
    custom_configuration = InferenceConfiguration(confidence_threshold=0.01)
    CLIENT.configure(custom_configuration)
    
    for patch_file in patch_folder:
        patch_path = os.path.join("static", patch_file)
        # img = Image.open(patch_path)
        # # Apply instance segmentation model here (e.g., U-Net)
        # # Save the segmented image
        # segmented_path = os.path.join(segmented_folder, patch_file)
        # img.save(segmented_path)  # Replace with actual segmented image
        # segmented_patches.append(segmented_path)
         # Initialize the client
        # Infer on a local image
        result = CLIENT.infer(patch_file, model_id="plantation-detection/2")

        # Extract bounding box information from the result
        # Assuming the result contains a list of predictions with bounding boxes
        predictions = result['predictions']

        # Load the image
        img = cv2.imread(patch_file)

        # Extract and save the portions inside the bounding boxes
        for idx, pred in enumerate(predictions):
            x = int(pred['x'])
            y = int(pred['y'])
            width = int(pred['width'])
            height = int(pred['height'])

            top_left_x = x - width // 2
            top_left_y = y - height // 2
            bottom_right_x = x + width // 2
            bottom_right_y = y + height // 2

            # Extract the portion of the image
            extracted_img = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            # Save the extracted image
            cv2.imwrite(os.path.join(segmented_folder, f"{tag}tree_{idx}.png"), extracted_img)
            if os.path.join(segmented_folder, f"{tag}tree_{idx}.png") != '':
                segmented_patches.append(os.path.join(segmented_folder, f"{tag}tree_{idx}.png"))

        print("Extracted images saved successfully!")
        tag = tag+1

    return segmented_patches

def count_trees(segmented_folder, counted_folder):
    # Dummy tree counting logic
    patch_files = os.listdir(segmented_folder)
    counts = {}
    # model = main.deepforest()
    # model.use_release()
    custom_configuration = InferenceConfiguration(confidence_threshold=0,)
    CLIENT = InferenceHTTPClient(
      api_url="https://detect.roboflow.com",
      api_key="QylQHoC503nypJSvSH3h"
    )
    CLIENT.configure(custom_configuration)
    
    
    for patch_file in patch_files:
        patch_path = os.path.join('E:/MiniProject3/static/segmented', patch_file)
        # img = Image.open(patch_path)
        # image_path = get_data(patch_path)
        # boxes = model.predict_image(path=image_path, return_plot = False)
        result = CLIENT.infer(patch_path, model_id="tree-count-c5uqe/1")
        # Apply tree counting model (e.g., CNN)
        # try:
        #     tree_count = len(boxes)  # Replace with actual tree count
        #     counts[patch_file] = tree_count
        #     img = model.predict_image(path=image_path, return_plot=True)
        #     # Save the counted image (Optional visualization)
        #     counted_path = os.path.join(counted_folder, patch_file)
        #     plt.imsave(counted_path,img[:,:,::-1])
        # except TypeError:
        #     print(f"Error processing {patch_file}: No predictions found")
        counted_path = os.path.join(counted_folder, patch_file)
        try:
            counts[patch_file] = len(result['predictions'])
        except TypeError:
            print(f"Error processing {patch_file}: No predictions found")
    return counts
