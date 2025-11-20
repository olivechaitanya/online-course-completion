import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config first
st.set_page_config(page_title="Online Course Completion Predictor", 
                  page_icon="🎓",
                  layout="wide")

# Initialize model paths
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaler.pkl')

# Load or train model
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.sidebar.success('Model loaded successfully!')
except:
    st.sidebar.info('Training model...')
    
    # Load and preprocess data
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'olive- online course completion.csv'))
    df = df.drop('UserID', axis=1)
    df = pd.get_dummies(df, columns=['CourseCategory'])
    
    # Train model
    X = df.drop(['CourseCompletion', 'CompletionRate'], axis=1)
    y = df['CourseCompletion']
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    
    # Save model
    joblib.dump(model, model_path)
    joblib.dump(StandardScaler().fit(X), scaler_path)
    
    st.sidebar.success('Model trained and saved successfully!')

# UI
st.title("🎓 Online Course Completion Predictor")

with st.form('course_form'):
    col1, col2 = st.columns(2)
    
    with col1:
        course_category = st.selectbox("Course Category", ["Health", "Arts", "Science", "Programming", "Business"])
        time_spent = st.slider("Time Spent (hours)", 0.0, 100.0, 30.0, 0.1)
        videos_watched = st.slider("Videos Watched", 0, 20, 10)
    
    with col2:
        quizzes_taken = st.slider("Quizzes Taken", 0, 15, 5)
        quiz_scores = st.slider("Quiz Scores", 0.0, 100.0, 75.0, 0.1)
        device_type = st.selectbox("Device", ["Mobile (1)", "Desktop (0)"])
    
    submit = st.form_submit_button('Predict')

if submit:
    # Prepare input
    input_data = {
        'TimeSpentOnCourse': [time_spent],
        'NumberOfVideosWatched': [videos_watched],
        'NumberOfQuizzesTaken': [quizzes_taken],
        'QuizScores': [quiz_scores],
        'DeviceType': [1 if device_type == "Mobile (1)" else 0]
    }
    
    # Add categories
    categories = ['CourseCategory_Arts', 'CourseCategory_Business', 'CourseCategory_Health', 
                 'CourseCategory_Programming', 'CourseCategory_Science']
    
    # Ensure all categories are present in input data
    for cat in categories:
        input_data[cat] = [1 if cat.split('_')[-1] == course_category else 0]
    
    # Predict
    input_df = pd.DataFrame(input_data)
    
    # Ensure all features match the training data
    all_features = ['TimeSpentOnCourse', 'NumberOfVideosWatched', 'NumberOfQuizzesTaken',
                   'QuizScores', 'DeviceType', 'CourseCategory_Arts', 'CourseCategory_Business',
                   'CourseCategory_Health', 'CourseCategory_Programming', 'CourseCategory_Science']
    
    # Reorder columns to match training data
    input_df = input_df[all_features]
    
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1] * 100
    
    # Show results
    if prediction == 1:
        st.success(f"✅ Likely to complete! (Confidence: {proba:.1f}%)")
    else:
        st.error(f"❌ Unlikely to complete (Confidence: {proba:.1f}%)")


# Add some visualizations
st.subheader("Data Analysis Visualizations")

# Create sample data for visualization
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=['Health', 'Arts', 'Science', 'Programming', 'Business'],
           y=[0.65, 0.55, 0.75, 0.85, 0.60],
           ax=ax)
ax.set_title('Course Completion Rate by Category')
ax.set_ylabel('Completion Rate')

# Show the plot
st.pyplot(fig)

# Add feature importance visualization
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': ['TimeSpentOnCourse', 'NumberOfVideosWatched', 'NumberOfQuizzesTaken',
               'QuizScores', 'DeviceType', 'CourseCategory_Arts', 'CourseCategory_Business',
               'CourseCategory_Health', 'CourseCategory_Programming', 'CourseCategory_Science'],
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# Create feature importance plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
ax.set_title('Feature Importance')
st.pyplot(fig)

# Add sidebar with project information
st.sidebar.title("About")
st.sidebar.info("""
This application predicts the likelihood of students completing online courses based on their engagement metrics.

Features considered:
- Course Category
- Time Spent on Course
- Number of Videos Watched
- Number of Quizzes Taken
- Quiz Scores
- Device Type
""")

# Add footer
st.markdown("---")
st.write("Created by Olive chaitanya maddirala")
