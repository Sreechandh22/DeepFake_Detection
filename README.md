# DeepFake Detection: Identifying Real vs. Fake Images Using AI

Welcome to the DeepFake Detection project repository. This project focuses on detecting deepfake images using advanced AI models. The solution leverages pre-trained models and custom CNN architectures to classify images as either real or fake. 

## Table of Contents

- [Introduction](#introduction)
- [Project Objectives](#project-objectives)
- [Technology and Tools](#technology-and-tools)
- [Setup](#setup)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Using the Model for Prediction](#using-the-model-for-prediction)
- [File Structure](#file-structure)
- [License](#license)
- [Contact](#contact)

## Introduction

DeepFake Detection is a project aimed at distinguishing between real and deepfake images using deep learning techniques. The project utilizes models like MobileNetV2 and Convolutional Neural Networks (CNNs) for accurate image classification. It addresses the growing concern of misinformation caused by deepfake content in media.

## Project Objectives

1. Implement AI techniques to detect deepfake images.
2. Use MobileNetV2 and CNNs to classify images as real or fake.
3. Achieve high accuracy in detecting manipulated media to mitigate the effects of fake content.

## Technology and Tools

- **Frameworks**: TensorFlow, Keras
- **Pre-trained Models**: MobileNetV2
- **Languages**: Python
- **Libraries**: NumPy, Pandas, Matplotlib, Streamlit
- **Deployment**: Streamlit, Google Colab

## Setup

### 1. Clone the repository

```sh
git clone https://github.com/Sreechandh22/DeepFake_Detection.git
cd DeepFake_Detection
```
### 2. Create a virtual environment and activate it

For Windows:

```sh
python -m venv venv
venv\Scripts\activate
```

For Linux/Mac:

```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the required packages

```sh
pip install -r requirements.txt
```

### 4. Download the dataset (if not already available)

This project uses the Deepfake and Real Images Dataset. Ensure that the dataset is available in the Dataset directory and organized into Train, Validation, and Test folders.

You can find the dataset on Kaggle.


## Usage
  Training the Model

  
  Prepare the data:
  
  Ensure that the dataset is structured as follows:
    
      Dataset/Train/ for training images
  
      Dataset/Validation/ for validation images
      
      Dataset/Test/ for testing images
      
  
  Run the training script:

  Use the following command to train the model:
  
      python app.py


## Using the Model for Prediction
  
  Load the Model and Predict:

  You can load the trained model and use it to predict whether an uploaded image is real or fake. Here's an example:
   
  ```sh
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import numpy as np
    
    # Load model
    model = load_model('deepfake_detector_model.h5')
    
    # Load an image for prediction
    img = image.load_img('path_to_image.jpg', target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Predict
    prediction = model.predict(img_array)
    print('Real' if prediction < 0.5 else 'Fake')
```

Deploy the App:

To deploy the app using Streamlit, run:

 ```sh
    streamlit run app.py
 ```

## File Structure

    DeepFake_Detection/
      ├── Dataset/
      │   ├── Train/
      │   ├── Validation/
      │   ├── Test/
      ├── README.md
      ├── app.py
      ├── requirements.txt
      ├── deepfake_detector_model.h5

## License

This project is licensed under the MIT License.

## Contact

For any inquiries or collaboration opportunities, please contact sreechandh2204@gmail.com.









