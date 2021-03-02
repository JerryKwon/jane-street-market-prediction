# jane-street-market-prediction

**Coworking of UNIST SDMLAB**.

<a href="https://github.com/JerryKwon">Youngin Kwon</a>, <a href="https://github.com/YeongHo-Lee">Yeongho Lee</a>, <a href="https://github.com/kcsayem">MD KHALEQUZZAMAN CHOWDHURY SAYEM</a>, <a href="https://github.com/tombstone013">MUBARRAT CHOWDHURY</a> 

##  :triangular_flag_on_post: Competition info

### :label: ​Name

<a href="https://www.kaggle.com/c/jane-street-market-prediction">Jane Street Market Prediction</a> on Kaggle

### :mag: Objective

Make a Prediction for trading action using trading opportunities

### :stopwatch: Timeline

Nov 24, 2020 - Feb 22 2021 (UTC)

**Actual participation, Jan 13 2021**

### :spiral_calendar: ​Overall Schedule

* 1st week: Understanding about Competition with EDA, Implementation using baseline code per solutions
  * Youngin: LSTM 
  * Yeongho: XGBoost
  * Sayem: Comprehensive Presentation
  * Mubarrat: LGBM
  
* 2nd week: [Enhancing Performance] Implementation of baseline code individually like above

* 3rd week: Best score solution Implementation

* 4th week: [Enhancing Performance] Best score solution Implementation
  
  * Youngin: Resnet1dcnn, ResnetLinear, EmbedNN
  
* 5th week: Parameter tuning for Best score solution

  **Resnet1dcnn**

  | layers  | hidden_layers | f_act | dropout | optimizer | learning_rate | weight_decay |
  | ------- | ------------- | ----- | ------- | --------- | ------------- | ------------ |
  | [5,5,5] | [512,128]     | ReLU  | 0.2     | Adam      | 1e-3          | 1e-5         |

  **ResnetLinear**

  | elected | Mean AUC | Mean Utility | Fold1-AUC | Fold1-Utility | Fold2-AUC | Fold2-Utility | Fold3-AUC | Fold3-Utility | hidden-layer | n_layers | decreasing | f_act     | dropout             | embed_dim | optimizer | learning_rate          | weight_decay           |
  | ------- | -------- | ------------ | --------- | ------------- | --------- | ------------- | --------- | ------------- | ------------ | -------- | ---------- | --------- | ------------------- | --------- | --------- | ---------------------- | ---------------------- |
  | False   | 0.5309   | 6162.2528    | 0.5299    | 6433.2630     | 0.5313    | 5603.7364     | 0.5315    | 6449.7590     | 512          | 3        | True       | LeakyReLU | 0.34213845887711536 | 10        | Adam      | 0.0009437366580626903  | 1.0288953711004482e-08 |
  | True    | 0.5308   | 6433.8982    | 0.5301    | 6495.6230     | 0.5319    | 6123.3656     | 0.5305    | 6682.7062     | 256          | 2        | False      | SiLU      | 0.49627361377205387 | 0         | Adam      | 1.3352033297894747e-05 | 8.62843672831598e-08   |

  **EmbedNN**

  | Selected | Mean AUC | Mean Utility | Fold1-AUC | Fold1-Utility | Fold2-AUC | Fold2-Utility | Fold3-AUC | Fold3-Utility | hidden-layer | n_layers | decreasing | f_act | dropout             | embed_dim | optimizer | learning_rate          | weight_decay           |
  | -------- | -------- | ------------ | --------- | ------------- | --------- | ------------- | --------- | ------------- | ------------ | -------- | ---------- | ----- | ------------------- | --------- | --------- | ---------------------- | ---------------------- |
  | False    | 0.5311   | 6058.2003    | 0.5323    | 6253.5662     | 0.5307    | 5494.9595     | 0.5303    | 6426.0753     | 256          | 3        | False      | SiLU  | 0.23308511537027937 | 10        | Adam      | 0.000663767918321238   | 2.6504094565959894e-07 |
  | True     | 0.5326   | 6055.6161    | 0.5322    | 5822.0162     | 0.5320    | 5811.6806     | 0.5338    | 6533.1516     | 256          | 4        | True       | SiLU  | 0.17971171427796284 | 5         | Adam      | 2.9521544108896628e-05 | 5.679142529741758e-05  |

  

## :loudspeaker: ​Repository Rule 

### :construction_worker: Structure

```
+-- input
|   +-- data/competitions/jane-street-market-prediction (competition dataset directory)
|   +-- kaggle.json (need for using kaggle API installing competition dataset)
+-- ipynb_notebooks (member's local code directory) 
|   +-- youngin
|   +-- yeongho
|   +-- sayem
|   +-- mubarrat
+-- output
|   +-- model (save pretrained model for inference)
|   +-- result (save .csv result file after inference)
+-- imgs (imgs for repository)
+-- README.md
+-- data_loader.py (methods for loading data)
+-- data_utils.py (methods for data preprocessing / model training / model inference)
+-- models.py (classes for Neural Net models)
+-- best_params.py (classes for storing best parameters)
+-- cv.py (classes for splitting Time-Series Aware Cross Validation method)
+-- train.py (python script for executing training process)
+-- inference.py (python script for executing inference process)

```

### :palm_tree: ​Branches 

Branch is tool of github for cooperation.

\- **for handling admission to master branch, efficient version management (need to find how to use)**

*  jerry: Youngin-Kwon
* ho: Yeongho-Lee
* sayem: MD KHALEQUZZAMAN CHOWDHURY SAYEM
* mubart: MUBARRAT CHOWDHURY

## :wrench: Installed Packages

|name|version|
|----|----|
|datatable|0.11.1|
|kaggle|1.5.10|
|numba|0.51.2|
|numpy|1.20.1|
|pandas|1.2.2|
|pytorch|1.7.1|
|scikit-learn|0.24.1|
|zipfile38|0.0.3|

## :electric_plug: ​Implementation

### train.py

* options
  
  * --model_type [Resnet1dcnn, ResnetLinear, EmbedNN]
  
    : define Neural Net model for training
  
  * --cv_type [SGCV, GTCV, random**(Resnet1dcnn only)**]
  
    : define Cross Validation Type for dataset
  
  * --selection [1,2] **(ResnetLinear, EmbedNN only)**
  
    : define best parameter 

### inference.py

* options
  * --model_type [Resnet1dcnn, ResnetLinear, EmbedNN]

    : define Neural Net model for training

  * --submit_type [csv, package**(Linux only)**]

    : define submission type

  * --cv_type [SGCV, GTCV, random**(Resnet1dcnn only)**]

    : define Cross Validation Type for dataset

  * --selection [1,2] **(ResnetLinear, EmbedNN only)**

    : define best parameter 

  * --pretrained [True, False]

    : define use pretrained weight for inference

### Examples

* train Resnet1dcnn model

  python train.py --model_type Resnet1dcnn --cv_type random

* train ResnetLinear model

  python train.py --model_type ResnetLinear --cv_type SGCV --selection 1

* (no trained .pth file) inference Resnet1dcnn model

  python inference.py --model_type Resnet1dcnn --submit_type csv --cv_type random --pretraind False

* (There is trained .pth file) inference Resnet1dcnn model

  python inference.py --model_type Resnet1dcnn --submit_type csv --cv_type random --pretraind True