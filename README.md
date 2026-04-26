# ML_Project
NutriScan: An Adaptive Machine Learning Framework for Precision Nutrition and Metabolic Management.

Features
  RGB-based food analysis
  RGB-D multimodal learning
  Calorie prediction
  Macronutrient prediction
  Monocular depth estimation using MiDaS
  3D depth visualization
  Relative food-volume analysis
  Nutrition5k dataset integration

  Dataset

This project uses the Nutrition5k dataset.

Dataset includes:

RGB food images
Depth maps
Calories
Food mass
Fat
Carbohydrates
Protein

**Dataset Link:**
https://github.com/google-research-datasets/Nutrition5k


Technologies Used
Python
PyTorch
Torchvision
OpenCV
NumPy
Pandas
Matplotlib
MiDaS


**Installation**

Install required packages:
Can install from requirements.txt or can use the below command.

pip install torch torchvision pandas numpy opencv-python matplotlib pillow timm


**Train RGB + Depth Model
python train_depth.py**

Run INference
with monocular depth:
python infer_with_mono.py

with out monocular depth:
python infer_without_mono.py


Methodology

The system uses:

RGB feature extraction using ResNet18
Depth feature extraction using CNN layers
Feature fusion
Regression layers for nutrition prediction

The project also integrates MiDaS for monocular depth estimation from single RGB images.

Outputs

The system predicts:

Calories
Food Mass
Fat
Carbohydrates
Protein

It also generates:

  Depth maps
  3D surface visualizations
  Relative volume estimations
  Future Work
  Food segmentation
  Accurate 3D food-volume estimation
  USDA nutritional mapping
  Mobile application deployment
  Personalized dietary recommendation system

  
**Authors**
Sri Varsha Maram – U01131261 
Sri Charan Vutukuru – U01113789
