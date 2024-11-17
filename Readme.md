# Hand Gesture Recognition using Small HGR1 Dataset

## Motivation
Last year, I visited a charity school in Vietnam for children who are deaf and have disabilities. I wanted to find a way to connect with them, and I thought that sign language could be a solution. While there are many sign language recognition models out there, I decided to create my own. Despite not having access to a powerful PC cluster or a large dataset, I aimed to improve accuracy using a small dataset to see how effective it can be.

## This Project
In this project, I use my experience in Computer Vision and Data Science to push the limits of a small dataset and explore how much accuracy can be achieved. I am using ResNet50 for its excellent performance and transfer learning to improve the model's effectiveness. The results are noticeable when comparing the 99% accurate ResNet50 model to the 96% fine-tuned model.

## How to Run
Start by cloning the repository:

```bash
git clone https://github.com/lds217/Hand-Gesture-Recognition-using-small-HGR1-dataset.git
```

Create a virtual environment with Python **3.8 - 3.10**, and then install the required dependencies:

**For training:**
```bash
pip install -r requirements-train.txt
```
**For runing:**
```bash
pip install -r requirements-run.txt
```

**SHOULD USE DIFFERENT VENV FOR DIFFERENT PURPOSE!**

For GPU acceleration, install CUDA Toolkit 11.2 and cuDNN 8.1.0. To install, download the CUDA Toolkit (exe) and cuDNN (zip), then extract and move the three cuDNN folders to the toolkit directory, replacing the existing files.
**And remember to downgrade tensorflow to 2.10.1** if you want to use GPU

Next, download the original images and skin masks from the following links:
- [Original Images](https://sun.aei.polsl.pl/~mkawulok/gestures/hgr1_images.zip)
- [Skin Masks](https://sun.aei.polsl.pl/~mkawulok/gestures/hgr1_skin.zip)

Structure your project folder like this (for training):

```
Hand-Gesture-Recognition-using-small-HGR1-dataset/
├── Data/
│   ├── original_images/
│   └── skin_masks/
├── venv/               # Virtual environment for dependencies
├── label.txt
├── requirements.txt
└── gesture_recognize.ipynb
```

### Run the Notebook
1. Run the "Import libraries" cell to check for dependencies.
2. Run the "Apply mask to hands and separate into folders" task to extract the hands from images.
3. Continue with training or explore the app functionality.

### Run the Webcam App
For the app you don't need to do anything else!

Make sure you install all the dependencies in `requirements-run.txt` and then just run `app.py`.

You can also download from Hugging_face with this [link](https://huggingface.co/lds217/HGR1-resnet50-transferlearning/tree/main).

Please be patient because it will take multiple times on weak machines to run it. If I have more time I would implement a framerate limiter so it can adapt every devices

## Conclusion
This project was a great exercise in working with a small dataset. Different approaches resulted in varying levels of accuracy, and the project can be further modified and improved.

