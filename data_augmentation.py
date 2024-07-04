import os
import cv2
import random
import imgaug as ia
import imgaug.augmenters as iaa


def augment_rotate(image, bbs):
    rotate_angle = random.uniform(-45, 45)
    seq = iaa.Sequential([iaa.Affine(rotate=rotate_angle)])
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
    return image_aug, bbs_aug


def augment_translate(image, bbs):
    translate_x = random.uniform(-0.2, 0.2)
    translate_y = random.uniform(-0.2, 0.2)
    seq = iaa.Sequential([iaa.Affine(translate_percent={"x": translate_x, "y": translate_y})])
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
    return image_aug, bbs_aug


def augment_flip(image, bbs):
    seq = iaa.Sequential([iaa.Fliplr(random.uniform(0, 1)), iaa.Flipud(random.uniform(0, 1))])
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
    return image_aug, bbs_aug


def augment_crop(image, bbs):
    crop_percent = random.uniform(0, 0.1)
    seq = iaa.Sequential([iaa.Crop(percent=crop_percent)])
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
    return image_aug, bbs_aug


def augment_stretch(image, bbs):
    stretch_scale_x = random.uniform(0.8, 1.2)
    stretch_scale_y = random.uniform(0.8, 1.2)
    seq = iaa.Sequential([iaa.Affine(scale={"x": stretch_scale_x, "y": stretch_scale_y})])
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
    return image_aug, bbs_aug


def apply_random_augmentations(image, bbs):
    augmentations = [augment_rotate, augment_translate, augment_flip, augment_crop, augment_stretch]
    num_augmentations = random.randint(1, len(augmentations))
    selected_augmentations = random.sample(augmentations, num_augmentations)

    for aug in selected_augmentations:
        image, bbs = aug(image, bbs)
    return image, bbs


def load_images_and_annotations(image_dir, annotation_dir):
    images = []
    bbs_list = []
    filenames = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            txt_path = os.path.join(annotation_dir, os.path.splitext(filename)[0] + '.txt')
            if os.path.exists(txt_path):
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Unable to read image {img_path}")
                    continue
                bbs = []
                with open(txt_path, 'r') as file:
                    for line in file.readlines():
                        data = line.strip().split()
                        class_id = int(data[0])
                        bbox = [float(x) for x in data[1:]]
                        x_center, y_center, width, height = bbox
                        bbs.append(ia.BoundingBox(
                            x1=(x_center - width / 2) * img.shape[1],
                            y1=(y_center - height / 2) * img.shape[0],
                            x2=(x_center + width / 2) * img.shape[1],
                            y2=(y_center + height / 2) * img.shape[0],
                            label=class_id
                        ))
                images.append(img)
                bbs_list.append(bbs)
                filenames.append(filename)
    return images, bbs_list, filenames


def save_augmented_image_and_annotations(image, bbs, output_image_dir, output_annotation_dir, count):
    augmented_img_path = os.path.join(output_image_dir, f"{count}.png")
    cv2.imwrite(augmented_img_path, image)

    txt_path = os.path.join(output_annotation_dir, f"{count}.txt")
    with open(txt_path, 'w') as file:
        for bb in bbs:
            x_center = ((bb.x1 + bb.x2) / 2) / image.shape[1]
            y_center = ((bb.y1 + bb.y2) / 2) / image.shape[0]
            width = (bb.x2 - bb.x1) / image.shape[1]
            height = (bb.y2 - bb.y1) / image.shape[0]
            file.write(f"{bb.label} {x_center} {y_center} {width} {height}\n")


def augment_until_target_images(image_dir, annotation_dir, output_image_dir, output_annotation_dir, target_count):
    images, bbs_list, filenames = load_images_and_annotations(image_dir, annotation_dir)
    existing_images_count = len(images)

    if existing_images_count == 0:
        print("No images found in the input directory.")
        return

    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    if not os.path.exists(output_annotation_dir):
        os.makedirs(output_annotation_dir)

    augmented_images_count = existing_images_count

    image_indices = list(range(existing_images_count))
    random.shuffle(image_indices)

    while augmented_images_count < target_count:
        if not image_indices:
            image_indices = list(range(existing_images_count))
            random.shuffle(image_indices)

        index = image_indices.pop()
        img = images[index]
        bbs = ia.BoundingBoxesOnImage(bbs_list[index], shape=img.shape)

        augmented_image, augmented_bbs = apply_random_augmentations(img, bbs)

        augmented_images_count += 1
        save_augmented_image_and_annotations(augmented_image, augmented_bbs.bounding_boxes, output_image_dir,
                                             output_annotation_dir, augmented_images_count)

        print(f"\rAugmented {augmented_images_count} images out of {target_count}", end="")

    print(f"\nTotal images after augmentation: {augmented_images_count}")


def process_dataset(dataset_dir, target_count):
    for class_folder in os.listdir(dataset_dir):
        class_folder_path = os.path.join(dataset_dir, class_folder)
        if os.path.isdir(class_folder_path):
            image_dir = os.path.join(class_folder_path, 'Images')
            annotation_dir = os.path.join(class_folder_path, 'Labels')

            if os.path.exists(image_dir) and os.path.exists(annotation_dir):
                image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
                image_count = len(image_files)

                if image_count > target_count:
                    print(f"Trimming class folder: {class_folder} from {image_count} to {target_count} images")
                    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]), reverse=True)
                    for i in range(image_count - target_count):
                        img_to_remove = image_files[i]
                        os.remove(os.path.join(image_dir, img_to_remove))
                        annotation_to_remove = os.path.splitext(img_to_remove)[0] + '.txt'
                        if os.path.exists(os.path.join(annotation_dir, annotation_to_remove)):
                            os.remove(os.path.join(annotation_dir, annotation_to_remove))
                    image_count = target_count

                if image_count < target_count:
                    print(f"Processing class folder: {class_folder}")
                    augment_until_target_images(image_dir, annotation_dir, image_dir, annotation_dir, target_count)
                else:
                    print(f"Class folder: {class_folder} already has {target_count} images")


dataset_directory = r"dataset"
process_dataset(dataset_directory, 2000)
