# Online Course Completion Rate Prediction App

This is a Streamlit web application that predicts the likelihood of students completing online courses based on their engagement metrics.

## Features

- Predict course completion probability
- Visualize completion rates by course category
- Show feature importance
- User-friendly interface for inputting student data
- Real-time predictions with confidence scores

## Installation

1. Install the required packages:
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

2. Run the application:
```bash
streamlit run course_completion_app.py
```

## How to Use

1. Open the application in your web browser
2. Fill in the student details:
   - Course Category
   - Time Spent on Course
   - Number of Videos Watched
   - Number of Quizzes Taken
   - Quiz Scores
   - Device Type
3. Click "Predict" to get the completion probability
4. View the feature importance chart to understand which factors are most influential

## Project Structure

- `course_completion_app.py`: Main Streamlit application file
- `model.pkl`: Trained machine learning model
- `scaler.pkl`: Data preprocessing scaler
- `olive- online course completion.csv`: Dataset used for training

## Model Details

- Algorithm: Random Forest Classifier
- Features:
  - TimeSpentOnCourse
  - NumberOfVideosWatched
  - NumberOfQuizzesTaken
  - QuizScores
  - DeviceType
  - CourseCategory (one-hot encoded)

## Created By

Olive Chaitanya Maddirala
