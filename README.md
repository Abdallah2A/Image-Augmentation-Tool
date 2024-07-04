# Image Augmentation Tool

## Description
This Python script performs image augmentation on a dataset of images and their corresponding annotations using imgaug library. It applies various transformations such as rotation, translation, flipping, cropping, and stretching to generate augmented images for training computer vision models.

## Requirements

- Python 3.x

- OpenCV (pip install opencv-python)

- imgaug (pip install imgaug)

## Installation

Clone the repository:
```bash
git clone https://github.com/Abdallah2A/Image-Augmentation-Tool.git
```

## Detailed Steps
### Augmentation Functions
- Rotate: Random rotation between -45° to 45°.

- Translate: Random translation by up to 20% in both x and y directions.

- Flip: Random horizontal and vertical flips.

- Crop: Random cropping by up to 10%.

- Stretch: Random scaling by 80% to 120% in both x and y directions.

### Dataset Handling
- Load Images and Annotations: Reads images and corresponding bounding box annotations from directories.

- Save Augmented Images and Annotations: Saves augmented images and updated annotations to output directories.

- Augmentation Process

- Randomly selects images from the dataset.

- Applies a random combination of augmentations to each selected image until the target number of images is reached.
