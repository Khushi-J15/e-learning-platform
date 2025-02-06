import streamlit as st
import pickle
import gzip
import pandas as pd
import neattext.functions as nfx
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained model from gzip-compressed pickle file
@st.cache_data
def load_model():
    with gzip.open('recommendation_model.pkl.gz', 'rb') as file:
        count_vect, cv_mat, cosine_sim_mat, df = pickle.load(file)
    return count_vect, cv_mat, cosine_sim_mat, df

count_vect, cv_mat, cosine_sim_mat, df = load_model()

# Function to recommend courses
def recommend_courses(course_name, num_recommendations=6):
    # Find courses containing the keyword
    relevant_courses = df[df['clean_course_title'].str.contains(course_name, case=False, na=False)]
    
    if not relevant_courses.empty:
        return relevant_courses['course_title'].head(num_recommendations).tolist()  # Return only top 6 matching courses
    
    # Clean the entered course name
    course_name = nfx.remove_stopwords(course_name)
    course_name = nfx.remove_special_characters(course_name).lower().strip()

    # Check if the course exists in the dataset
    if course_name not in df['clean_course_title'].values:
        return ["No exact match found. Try a similar keyword!"]
    
    # Get the course index
    course_index = df[df['clean_course_title'] == course_name].index[0]

    # Get similarity scores
    similarity_scores = list(enumerate(cosine_sim_mat[course_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]  # Only top 6 recommendations

    recommended_courses = [df.iloc[i[0]]['course_title'] for i in similarity_scores]
    
    return recommended_courses

# ---- STREAMLIT UI ----

# Sidebar
st.sidebar.title("🔍 Course Recommender")
st.sidebar.write("Search and discover the best courses for your learning!")

# Main Dashboard
st.title("📚 E-Learning Platform")
st.subheader("Find the best courses tailored for you!")

# Search Bar
course_name = st.text_input("Enter a course keyword (e.g., 'Python', 'Machine Learning'):")

# Recommend Button
if st.button("🔎 Get Recommendations"):
    if course_name.strip():
        recommendations = recommend_courses(course_name)
        st.subheader("🎯 Recommended Courses:")
        for idx, rec in enumerate(recommendations, start=1):
            st.write(f"✅ {rec}")
    else:
        st.warning("⚠ Please enter a course name to get recommendations.")

# Footer
st.markdown("---")
st.markdown("🚀 Built with ❤️ using Streamlit")
