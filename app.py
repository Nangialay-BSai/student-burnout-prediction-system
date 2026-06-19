import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Student Burnout Prediction System",
    page_icon="🎓",
    layout="wide"
)

# =========================
# LOAD FILES
# =========================
df = pd.read_csv("ai_student_impact_dataset.csv")

model = joblib.load("burnout_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# =========================
# SIDEBAR
# =========================
page = st.sidebar.radio(
    "📌 Navigation",
    ["Dashboard", "Prediction", "Analytics", "About"]
)

# =========================
# DASHBOARD PAGE
# =========================
if page == "Dashboard":

    st.title("🎓 Student Burnout Prediction System")
    st.markdown("AI-Powered Student Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Students", len(df))

    with col2:
        st.metric(
            "Average GPA",
            round(df["Post_Semester_GPA"].mean(), 2)
        )

    with col3:
        st.metric(
            "Average AI Hours",
            round(df["Weekly_GenAI_Hours"].mean(), 2)
        )

    with col4:
        st.metric(
            "Average Anxiety",
            round(df["Anxiety_Level_During_Exams"].mean(), 2)
        )

    st.divider()

    st.subheader("Burnout Risk Distribution")

    burnout_count = (
        df["Burnout_Risk_Level"]
        .value_counts()
        .reset_index()
    )

    burnout_count.columns = ["Burnout", "Count"]

    fig = px.pie(
        burnout_count,
        names="Burnout",
        values="Count",
        title="Burnout Risk Levels"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# =========================
# PREDICTION PAGE
# =========================
elif page == "Prediction":

    st.title("🤖 Burnout Prediction")

    major = st.selectbox(
        "Major Category",
        df["Major_Category"].unique()
    )

    year = st.selectbox(
        "Year of Study",
        df["Year_of_Study"].unique()
    )

    gpa = st.number_input(
        "Pre Semester GPA",
        min_value=0.0,
        max_value=4.0,
        value=3.0
    )

    anxiety = st.slider(
        "Anxiety Level During Exams",
        1,
        10,
        5
    )

    if st.button("Predict Burnout Risk"):

        sample = pd.DataFrame({
            "Major_Category": [major],
            "Year_of_Study": [year],
            "Pre_Semester_GPA": [gpa],
            "Weekly_GenAI_Hours": [5.0],
            "Primary_Use_Case": ["Ideation"],
            "Prompt_Engineering_Skill": ["Intermediate"],
            "Tool_Diversity": [3],
            "Paid_Subscription": [False],
            "Traditional_Study_Hours": [10.0],
            "Perceived_AI_Dependency": [5],
            "Institutional_Policy": ["Allowed_With_Citation"],
            "Anxiety_Level_During_Exams": [anxiety],
            "Post_Semester_GPA": [gpa],
            "Skill_Retention_Score": [75]
        })

        encoded = preprocessor.transform(sample)

        prediction = model.predict(encoded)

        risk = prediction[0]

        if risk == "Low":
            st.success(f"🟢 Predicted Burnout Risk: {risk}")

        elif risk == "Medium":
            st.warning(f"🟡 Predicted Burnout Risk: {risk}")

        else:
            st.error(f"🔴 Predicted Burnout Risk: {risk}")

# =========================
# ANALYTICS PAGE
# =========================
elif page == "Analytics":

    st.title("📊 Analytics")

    major_chart = px.histogram(
        df,
        x="Major_Category",
        color="Burnout_Risk_Level",
        title="Burnout Risk by Major"
    )

    st.plotly_chart(
        major_chart,
        width="stretch"
    )

# =========================
# ABOUT PAGE
# =========================
elif page == "About":

    st.title("ℹ️ About Project")

    st.write("""
    ### Student Burnout Prediction System

    - Dataset Size: 50,000 Students
    - Machine Learning Model: Random Forest
    - Accuracy: 52.2%
    - Built With:
        - Python
        - Pandas
        - Scikit-Learn
        - Streamlit
        - Plotly
    """)