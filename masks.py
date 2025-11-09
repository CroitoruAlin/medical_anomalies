import cv2
import numpy as np
import random

def generate_random_blob_mask(image, num_blobs=5, min_size_factor=0.1, max_size_factor=0.4, blur_kernel_size=61):

    if image.ndim == 3:
        height, width, _ = image.shape
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        height, width = image.shape
        gray_image = image

    _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    non_black_coords = np.argwhere(thresh > 0)
    maxes = non_black_coords.max(axis=0)
    mins = non_black_coords.min(axis=0)

    height_mask = int(maxes[0]-mins[0])
    width_mask = int(maxes[1]-mins[1])
    if non_black_coords.shape[0] == 0:
        print("Warning: The provided image is entirely black. Returning an empty mask.")
        return np.zeros((height, width), dtype=np.uint8)

    mask = np.zeros((height, width), dtype=np.uint8)

    for _ in range(num_blobs):
        random_index = random.randint(0, non_black_coords.shape[0] - 1)
        y_center, x_center = non_black_coords[random_index]

        ax_1_len = int(random.uniform(min_size_factor, max_size_factor) * width_mask)
        ax_2_len = int(random.uniform(min_size_factor, max_size_factor) * height_mask)
        
        ax_1_len = max(1, ax_1_len // 2)
        ax_2_len = max(1, ax_2_len // 2)

        angle = random.randint(0, 179)

        cv2.ellipse(mask, (int(x_center), int(y_center)), (ax_1_len, ax_2_len), angle, 0, 360, (255), -1)


    if blur_kernel_size > 1:
        blur_kernel_size = int(blur_kernel_size) // 2 * 2 + 1
        mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask

# # --- Example Usage ---

# # 1. Create a sample image to act as our guide
# # This will be a black 512x512 image with a white rectangle (our "tissue region")
# img_height = 512
# img_width = 512
# test_image = np.zeros((img_height, img_width), dtype=np.uint8)
# # Define the "tissue" region
# test_image[100:400, 150:450] = 255 
# cv2.imwrite('guide_image.png', test_image)

# # 2. Generate the mask using the guide image
# # The blobs will now only be centered within the white rectangle
# blob_mask = generate_random_blob_mask(
#     test_image, 
#     num_blobs=10, 
#     min_size_factor=0.1,
#     max_size_factor=0.3,
#     blur_kernel_size=71
# )

# # 3. Save the final mask
# output_filename = 'constrained_blob_mask.png'
# cv2.imwrite(output_filename, blob_mask)

# print(f"Guide image saved as 'guide_image.png'")
# print(f"Constrained blob mask saved as '{output_filename}'")

# # 4. (Optional) Create an overlay to visualize the result
# test_image_color = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
# red_overlay = np.zeros_like(test_image_color)
# red_overlay[blob_mask == 255] = [0, 0, 255] # BGR for red
# overlay_image = cv2.addWeighted(test_image_color, 0.7, red_overlay, 0.3, 0)
# cv2.imwrite('mask_overlay.png', overlay_image)

# print(f"Visualization saved as 'mask_overlay.png'")