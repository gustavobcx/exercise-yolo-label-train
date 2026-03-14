# YOLO Label + Train (Bootcamp Practice)

A hands-on bootcamp project covering custom dataset creation, image labeling, and YOLO object detection training using transfer learning.

This project was created specifically for a bootcamp activity and to practice my knowledge. It is part of the [Coding The Future BairesDev - Machine Learning Practitioner bootcamp](https://www.dio.me/bootcamp/coding-the-future-baires-dev-machine-learning-practitioner).

## Original Challenge

> Project: create a dataset and train a YOLO network.
>
> Following the class examples, we will label a dataset and apply training with the YOLO network.
>
> For this task it will be necessary to use the LabelMe software to label the images.
>
> It will also be necessary to use the YOLO network.
>
> Those who prefer not to label a dataset from scratch may use the already-labeled COCO images.
>
> Those using a computer that cannot run the YOLO network locally may use transfer learning on Google Colab.
>
> The project must contain at least two newly trained classes for detection, in addition to the classes already trained before performing transfer learning.

Links referenced in the challenge:

- LabelMe: http://labelme.csail.mit.edu/Release3.0/
- YOLO (Darknet): https://pjreddie.com/darknet/yolo/
- COCO Dataset: https://cocodataset.org/#home
- Colab (transfer learning): https://colab.research.google.com/drive/1lTGZsfMaGUpBG4inDIQwIJVW476ibXk_#scrollTo=j0t221djS1Gk

## Project Goal

Train a YOLO model with 2 new custom classes:

- teapot
- mug

To satisfy the challenge requirements, the new classes were appended after the COCO classes (class ID offset of +80), preserving the transfer learning concept on top of a pre-trained model.

## Repository Structure

- `yolo_label_train.ipynb`: main notebook with the full pipeline.
- `custom-dataset/teapot-and-mug/`: LabelMe annotation files (.json).
- `runs/detect/train/`: training artifacts (metrics, weights, args).

Relevant output files:

- `runs/detect/train/weights/best.pt`
- `runs/detect/train/weights/last.pt`
- `runs/detect/train/results.csv`
- `runs/detect/train/args.yaml`

## Pipeline

1. Image labeling with LabelMe.
2. Conversion from LabelMe format to YOLO format using `labelme2yolo`.
3. Train/validation split (`val_size=0.3`).
4. New class ID adjustment with offset +80.
5. Model training via YOLO (Ultralytics) inside the notebook.

Commands used in the notebook (summary):

```bash
pip install labelme
labelme
pip install labelme2yolo
labelme2yolo --json_dir ./custom-dataset/teapot-and-mug --output_dir ./datasets/teapot-and-mug --val_size 0.3
```

## Training Configuration (recorded)

From `runs/detect/train/args.yaml`:

- model: yolov5su.pt
- data: datasets/coco128_plus2.yaml
- task: detect
- epochs: 20
- imgsz: 640
- batch: 16
- pretrained: true

## Results

Last recorded epoch (epoch 20, from `runs/detect/train/results.csv`):

| Metric | Value |
|---|---|
| precision(B) | 0.76494 |
| recall(B) | 0.61749 |
| mAP50(B) | 0.70307 |
| mAP50-95(B) | 0.55890 |

These results show consistent improvement across all 20 epochs for the dataset used in this exercise.

## How to Reproduce

### Option 1: Local (machine with compatible GPU/CPU)

1. Create and activate a virtual environment.
2. Open `yolo_label_train.ipynb`.
3. Run the cells in order:
   - dependency installation
   - labeling and conversion
   - class ID adjustment
   - training and evaluation

Example (Git Bash on Windows):

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -U pip
pip install jupyter labelme labelme2yolo ultralytics
jupyter notebook
```

### Option 2: Google Colab

If your machine cannot handle local training, run the workflow in the challenge's Colab notebook and adapt the paths to point to your dataset.

## Notes

- This repository is a study/practice project from the bootcamp.
- The notebook uses the Ultralytics YOLO ecosystem for training.
- The challenge references Darknet YOLO as a classical baseline; both approaches are valid for the educational purposes of this exercise.

## Further Reading & References

* **[Model Training with Ultralytics YOLO](https://docs.ultralytics.com/modes/train/)**: YOLO training documentation step by step and examples

* **[Train YOLOv5 on Custom Data](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/)**: YOLO training tutorial step by step and examples

* **[YOLOv5 Github repo](https://github.com/ultralytics/yolov5)**
