# Designing an Image Segmentation Model

An Image Segmentation Model designed to be used within the Computer Vision System 
of a Self Driving Vehicle.

It is trained based on the [Cityscrapes Dataset](https://www.cityscapes-dataset.com/).

This model identifies different classes of objects in photos captured by a Vehicle's sensors :

- Constructions
- Nature
- Sky
- People
- Vehicle
- Object

One of the challenge of this project was to reduce the computing power needed to train and deploy 
the model so it could be easily used by an edge device like the self-driving vehicle.

We used transfer learning and managed to reach state-of-the art performance on the Cityscrapes
dataset :

![State of the Art results](img/state_of_the_art_2.png)

The prediction API is then published on a web interface using Flask.

## Useful Links

* [Project presentation (Powerpoint)](Project_Presentation.pptx)
* [Technical Report of Model development (Word)](Technical_Report.docx)
* [Jupyter Notebook (Model training)](Notebook.ipynb)
* [Flask Deployment Folder](Deployment/)

## Screenshots

## Encoder-Decoder Framework
![Encoder-Decoder](img/encoder-decoder.png)

## Transfer Learning
![Transfer Learning](img/transfer_learning.png)

## Final Model Architecture
![Linknet](Linknet.png)

### Web Interface
![Web Interface](img/website.png)

### Model Prediction
![Raw Image](img/input.png)
![Prediction](img/prediction.png)
![True Mask](img/true.png)

### Predicts stock images
![Real Image](img/real_image.png)
![Prediction](img/real_prediction.png)

## Libraries / Packages Used

* [Tensorflow - Keras](https://www.tensorflow.org/)
* [Flask](https://flask.palletsprojects.com/en/2.2.x/)
* [Scikit-Image](https://scikit-image.org/)
* [Albumentations](https://albumentations.ai/)
* [Open CV](https://opencv.org/)
* [Segmentation Models](https://github.com/qubvel/segmentation_models)
* [SqueezeNet Keras Implementation](https://github.com/rcmalli/keras-squeezenet)
* [Bootstrap](https://getbootstrap.com/)
* Matplotlib / Seaborn
* Pandas / Numpy 

## Developed By

Octave Antoni

<img src="https://avatars.githubusercontent.com/u/841669?v=4" width="20%">

[Connect with me on Linkedin](https://www.linkedin.com/in/octave-antoni/)

## License

    Copyright 2023 Octave Antoni

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
