# infer.py

import os
import re
import json
import csv
import zipfile
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.amp import autocast
from torch.utils.data import Dataset
# from models import FasterRCNN_ResNet50_FPN
from new_models import FasterRCNN_ConvNeXt_FPN

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

IMAGE_FOLDER = "nycu-hw2-data/test"
WEIGHTS_PATH = "svhn_fasterrcnn_convnext_BEST.pth"
OUTPUT_JSON = "pred.json"
OUTPUT_CSV = "pred.csv"
OUTPUT_ZIP = "results.zip"
NUM_CLASSES = 11
BATCH_SIZE = 4
NUM_WORKERS = 4

JSON_SCORE_THRESHOLD = 0.0
CSV_SCORE_THRESHOLD = 0.7


def extract_number(filename):
    """
    Extracts the first number found in the filename.
    If no number is found, returns 0.
    """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0


def check(f):
    return f.lower().endswith(('.png', '.jpg', '.jpeg'))


class InferenceDataset(Dataset):
    def __init__(self, image_folder, transform):
        self.image_folder = image_folder
        # Get image files and sort them based on numeric value in filename
        self.image_files = [f for f in os.listdir(image_folder) if check(f)]
        self.image_files = sorted(self.image_files, key=extract_number)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        # Return image and its image_id, here we use idx+1 for image_id
        return image, idx + 1


def inference_collate(batch):
    # batch is a list of tuples (image, image_id)
    images, image_ids = zip(*batch)
    return list(images), list(image_ids)


def run_inference_batch(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Loads the model and runs inference on all images using batching.
    Returns a list of detection dictionaries.
    Each detection has keys: "image_id", "bbox", "score", and "category_id".
    Only detections with score >= SCORE_THRESHOLD are kept.
    """

    # Initialize model (pretrained set to False since we load custom weights)
    model = FasterRCNN_ConvNeXt_FPN(num_classes=NUM_CLASSES, pretrained=False)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = transforms.ToTensor()
    dataset = InferenceDataset(IMAGE_FOLDER, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=inference_collate
    )

    detections = []
    # Run inference with tqdm progress bar over batches
    with torch.no_grad():
        bar = tqdm(dataloader, desc="Running Inference Batch")
        for images, image_ids in bar:
            images = [img.to(device) for img in images]

            # Use autocast for CUDA if available
            enabled = (device.type == 'cuda')
            with autocast(device_type=device.type, enabled=enabled):
                outputs = model(images)

            # Process each image in the batch
            for image_id, output in zip(image_ids, outputs):
                boxes = output.get("boxes", [])
                scores = output.get("scores", [])
                labels = output.get("labels", [])

                zipper = zip(boxes.tolist(), scores.tolist(), labels.tolist())
                for box, score, label in zipper:
                    # Only keep detection if score >= SCORE_THRESHOLD
                    box[2] = box[2] - box[0]
                    box[3] = box[3] - box[1]
                    if score >= JSON_SCORE_THRESHOLD:
                        detection = {
                            "image_id": image_id,
                            "bbox": box,
                            "score": score,
                            "category_id": label
                        }
                        detections.append(detection)

    # Sort the detections by image_id (ascending order)
    detections.sort(key=lambda d: d["image_id"])
    return detections, len(dataset)


def save_json(detections):
    """
    Saves the detections list to a JSON file.
    """
    with open(OUTPUT_JSON, "w") as f:
        json.dump(detections, f, indent=4)
    print(f"JSON output saved to {OUTPUT_JSON}")


def generate_csv(detections, total_images):
    """
    Processes the detections to generate a CSV file.
    For each image, detections are sorted by x coordinate,
    then the (category_id - 1) values are concatenated as strings.
    If an image has no detections, pred_label is "-1".

    Args:
        detections: List of detection dictionaries.
        total_images: Total number of images processed.
    """
    # Group detections by image_id
    dets_by_img = {}
    for det in detections:
        img_id = det["image_id"]
        if img_id not in dets_by_img:
            dets_by_img[img_id] = []
        dets_by_img[img_id].append(det)

    rows = []
    # Loop over all image_ids
    for image_id in range(1, total_images + 1):
        dets = dets_by_img.get(image_id, [])

        # Filter detections by score threshold
        filtered = [d for d in dets if d["score"] >= CSV_SCORE_THRESHOLD]

        if filtered:
            # Sort by x coordinate of bbox
            sorted_dets = sorted(filtered, key=lambda d: d["bbox"][0])

            # Generate prediction label string
            pred_label = "".join(
                str(det["category_id"] - 1) for det in sorted_dets
            )
            rows.append([image_id, pred_label])
        else:
            rows.append([image_id, "-1"])

    # Write rows to CSV file
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "pred_label"])
        writer.writerows(rows)
    print(f"CSV output saved to {OUTPUT_CSV}")


def zip_outputs():
    """
    Creates a zip file containing the JSON and CSV outputs.
    """
    with zipfile.ZipFile(OUTPUT_ZIP, "w") as zipf:
        zipf.write(OUTPUT_JSON)
        zipf.write(OUTPUT_CSV)
    print(f"Outputs zipped into {OUTPUT_ZIP}")


def main():
    # Run batched inference and get detections
    # and the total number of images processed
    detections, total_images = run_inference_batch()

    # Save detections to JSON (sorted by image_id)
    save_json(detections)

    # Generate CSV file from the detections
    generate_csv(detections, total_images)

    # Zip both JSON and CSV outputs
    zip_outputs()


if __name__ == "__main__":
    main()
