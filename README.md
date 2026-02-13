# 📧 Spam Detection — End-to-End MLOps Pipeline (DVC • DVCLive • AWS)
![Spam Detection](https://img.shields.io/badge/Spam%20Detection-NLP%20Classification-e53935?style=for-the-badge&logo=gmail&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP%20%7C%20Classification-1e88e5?style=for-the-badge&logo=scikitlearn&logoColor=white)
![MLOps](https://img.shields.io/badge/MLOps-DVC%20%7C%20DVCLive-f57c00?style=for-the-badge&logo=databricks&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-S3%20Remote%20Storage-ffb300?style=for-the-badge&logo=amazonaws&logoColor=black)
![Python](https://img.shields.io/badge/Python-Scikit--Learn-1565c0?style=for-the-badge&logo=python&logoColor=ffeb3b)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%7C%20Text%20Processing-8e24aa?style=for-the-badge&logo=readthedocs&logoColor=white)
![Reproducibility](https://img.shields.io/badge/Reproducible-Yes-00c853?style=for-the-badge&logo=git&logoColor=white)
![Pipeline](https://img.shields.io/badge/Pipeline-DVC%20Repro-5e35b1?style=for-the-badge&logo=apacheairflow&logoColor=white)

<p align="center">
  <img src="Spam Detection.png" alt="Spam Detection MLOps Pipeline Poster" width="100%">
</p>



This project demonstrates an end-to-end MLOps pipeline for Spam Detection, covering the complete machine learning lifecycle from data ingestion to experiment tracking, with cloud-based artifact storage.

The project is designed to showcase industry-standard MLOps practices using DVC, DVCLive, and AWS S3 for data versioning, experiment tracking, and reproducibility.




## Project Description

Spam detection is a binary classification problem where messages are classified as Spam or Ham (Not Spam).

This project focuses on building a reproducible, scalable, and cloud-enabled machine learning pipeline rather than only achieving high model accuracy.

The main objectives of this project are:
- Build a modular end-to-end ML pipeline
- Track experiments and metrics using DVCLive
- Version datasets and models using DVC
- Store artifacts remotely using AWS S3
- Ensure reproducibility across different environments



## Project Structure

MLOps_Complete_End_to_End_ML_Pipeline  
├── .dvc/  
├── dvc.yaml  
├── dvc.lock  
├── params.yaml  
├── Experiments/  
├── dvclive/  
├── src/  
│   ├── data_ingestion.py  
│   ├── data_validation.py  
│   ├── data_transformation.py  
│   ├── model_trainer.py  
│   └── model_evaluation.py  
├── .gitignore  
├── .dvcignore  
├── projectflow.txt  
└── README.md  



## ML Pipeline Workflow

1. Data Ingestion  
   - Load raw spam dataset  
   - Store and version the dataset using DVC  

2. Data Validation  
   - Check missing values  
   - Validate schema and class distribution  

3. Data Transformation  
   - Text cleaning (lowercasing, punctuation removal)  
   - Tokenization and stopword removal  
   - Feature extraction using TF-IDF or CountVectorizer  

4. Model Training  
   - Train machine learning models such as Naive Bayes or Logistic Regression  
   - Hyperparameters are controlled via params.yaml  

5. Model Evaluation  
   - Evaluate the model using Accuracy, Precision, Recall, and F1-score  
   - Track metrics using DVCLive  



## Experiment Tracking

This project uses DVCLive for experiment tracking.

Tracked items include:
- Training and validation metrics
- Model performance across different experiments
- Logs and plots for comparison

All experiment-related artifacts are stored inside the dvclive directory.



## AWS Integration (DVC Remote Storage)

AWS S3 is used as a remote storage backend for DVC to store:
- Versioned datasets
- Trained models
- Intermediate pipeline artifacts

This enables scalable, durable, and cloud-based artifact management.

Example DVC remote configuration:

dvc remote add -d s3remote s3://<your-bucket-name>/dvc-store  
dvc push  

To retrieve data and models:
dvc pull  



## Parameter Management

All model and training parameters are centralized in the params.yaml file.

Example parameters include:
- Model type
- Hyperparameters
- Random state

This allows easy experiment tuning and clean versioning of parameters.



## Reproducibility

The project ensures full reproducibility using:
- dvc.yaml for pipeline definition
- dvc.lock for exact pipeline versions
- params.yaml for parameter tracking
- AWS S3 for remote artifact storage

Running the following command will reproduce the entire pipeline:
dvc repro







## Tech Stack

- Python
- Machine Learning
- Scikit-learn
- DVC
- DVCLive
- AWS S3
- NLP (TF-IDF / CountVectorizer)
- Git and GitHub
- YAML




## MLOps Concepts Demonstrated

- Data and model versioning
- Pipeline automation using DVC
- Experiment tracking with DVCLive
- Cloud-based artifact storage using AWS S3
- Parameterized ML pipelines
- Reproducible machine learning workflows



## Future Enhancements

- Model deployment using FastAPI
- MLflow model registry integration
- CI/CD using GitHub Actions
- Secure AWS access using IAM roles
- Deep learning-based spam detection models



## Author

Sanket  
Computer Engineering | Data Science | MLOps  
Focused on building scalable, reproducible, and cloud-native ML systems

