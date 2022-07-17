<div align="center">
    <img src=".\\miscellaneous\\station.png" alt="Weather station" width="800px" />
</div>

## Summary

Weather forecasting is an interesting research topic as changes in it could affect the environment around us as well as our day to day lives.
It is also a difficult thing to nail down, which is why we wanted to create a machine learning model with the use of neural network, turning it into
a relatively cheap and easy to deploy IoT device that could predict weather conditions with rather limited amount of data given through a sensor. With the use of MCU ESP32 and TFLite, it would be able to provide deep learning models through a repeatable and automated workflow. This project was done through a 3 month time-frame with rather limited parameters and budget. We wanted to show that it's possible to make a TF-Lite model with ANN that can tell or describe the weather of the current day and also able to tell the weather within the vicinity of the house's region. It will then show the results through the TF-Lite's output score where in each number will correspond to a weather type – whether it be cloudy, foggy, rainy or sunny. Though this project was able done in mentioned time-frame, we believe that there are still more room for improvement such as improving on feature engineering and its accuracy.

Have we tried it with RNN LSTM model?

- Yes, we actually have tried it and got a higher accuracy. Unfortunately, the current version of TFLite 1.1.2 wont allow the LSTM model to be converted so we had to compromise with an ANN model.
  <br><br>

## Dataset

The dataset used came from <a href="https://www.kaggle.com/">Kaggle</a>, an online community platform for data scientists and machine learning enthusiasts. It includes 96454 samples and 10 columns along with it. It has not been processed yet so it would be advisable to do some pre-processing before using it on the Neural network model.
<br><br>

## Model used

- Artificial Neural Network (CNN) with Keras tuning
  <br><br>

## Deployment

We have selected the best model by converting into TFLite and deploying it through ESP32 that will gather data through DHT-22 sensor and placed on a safe place outside the
vicinity of the house. With the data gathered, it will be stored to InfluxDB that will then be visualized on Grafana as a way to monitor it.

The weather station was created with a simple box in order to keep the sensor and breadboard safe from any harm.

<div align="center">
    <img src=".\\miscellaneous\\concept-design.jpg" alt="concept design" width="600px" /><br><br>
</div>

<div align="center">
    <img src=".\\miscellaneous\\actual.jpg" alt="actual" width="600px" /><br><br>
</div>
<br><br>

## Repository Contents

**dataset/**

- **weatherHistory.csv** - Original Kaggle weather dataset
  <br><br>

**miscellaneous/**

- **station.png** - Banner image for Github
- **concept-design.jpg** - Conceptual design of weather station
- **actual.jpg** - Actual weather station
  <br><br>

**models/** - Saved models from training and converting

- .h5 and .tflite files
  <br><br>

**platform.io** - Main C/C++ code for ESP32

- **include/** - README file for header
- **lib/** - README file for libraries
- **src/** - Main.cpp and .h file for compiling and running to the ESP32
- **platformio.ini** - Necessary configuration and dependencies for ESP32
  <br><br>

**main.ipynb** - Main Python code
<br><br>

**requirement.txt** - Necessary dependencies to make TFLite work
<br><br>

## Dependencies

Tested from April - May 2022

- Python - 3.7
- Tensorflow - 2.1.1
- Numpy - 1.22.0
- Pandas - 1.4.3
- Matplotlib - 3.5.1
- Scikit-learn - 1.0.2
- H5py - 2.10.0
- Seaborn - 0.11.2
  <br><br>

## Endnotes

We've run our code on Google Colab and Visual Studio Code with PlatformIO IDE, the files in this repository undergoes changes without testing so if there are any issues, please don't
hesitate to contact us.
<br><br>

## Contributors

1. **Joseph Leandre Derpo - https://github.com/LeandreDerpo**
2. **Chern-tay Shih - https://github.com/K-Tett**
3. **Nonthakorn Chencharatmatha - Haven't made one yet**
