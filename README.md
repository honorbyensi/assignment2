Diabetes Risk Predictor
This app predicts the likelihood of a person having diabetes, using the diabetes dataset from last week's class.
With this app, users can manually input data predictions, users should also be able to upload their own CSV file (but I couldnt get that working properly and it gave me an error, but the idea was for users to also have the option of upload their own CSV file to make predictions.)
The app also gives you access to the dataset vizualizations!
And to access the app you must be able to login in using a username and password.
Username: honor
Password: 2002

Model Description:
Model being used - Logistic Regression
Preprocessing Steps:
- Replaced zero values with NaN
- Median Imputation
- Standard Scaling
Dataset - Diabetes dataset provided in previous week
Train/Test split - 80% training / 20% testing
Test Accuracy - 70%

How to run the app locally:
1. Download repository
2. Install dependecies (pip install -r requirements.txt)
3. Run the app (streamlit run app.py)
4. The app should open at (http://localhost:8502)