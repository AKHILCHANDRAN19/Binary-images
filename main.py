import cv2
import os

def process_image(input_image_path, output_folder_path, resize_dimensions=(1020, 1020)):
    # Read the input image
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Failed to load image: {input_image_path}")
        return

    # Resize the image to the specified dimensions
    img_resized = cv2.resize(img, resize_dimensions)

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to convert the image to binary
    img_thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )

    # Ensure the output folder exists
    os.makedirs(output_folder_path, exist_ok=True)

    # Define the output image path
    output_image_path = os.path.join(output_folder_path, 'output_image.png')

    # Save the processed image
    cv2.imwrite(output_image_path, img_thresh)
    print(f"Processed image saved at: {output_image_path}")

if __name__ == "__main__":
    # Define the input image path
    input_image_path = "/storage/emulated/0/Download/images.jpeg"

    # Define the output folder path
    output_folder_path = "/storage/emulated/0/OUTPUT"

    # Process the image
    process_image(input_image_path, output_folder_path)
