import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Course Recommendation System", layout="wide")

@st.cache_data
def load_data():
    data = [
        {
            "title": "Python for Data Science",
            "description": "Learn Python basics for data analysis, data cleaning, and visualization using pandas and numpy.",
            "skills": "python,data analysis,pandas,numpy,data science",
            "level": "Beginner"
        },
        {
            "title": "Machine Learning Foundations",
            "description": "Introduction to supervised learning, unsupervised learning, model evaluation, and ML workflows.",
            "skills": "machine learning,python,statistics,supervised learning,unsupervised learning",
            "level": "Intermediate"
        },
        {
            "title": "Deep Learning Fundamentals",
            "description": "Learn neural networks, backpropagation, deep learning architectures, and real-world DL applications.",
            "skills": "deep learning,neural networks,python,ai",
            "level": "Advanced"
        },
        {
            "title": "SQL for Analytics",
            "description": "Use SQL to query data, analyze datasets, and solve business intelligence problems.",
            "skills": "sql,data analysis,database,business intelligence",
            "level": "Beginner"
        },
        {
            "title": "Natural Language Processing Basics",
            "description": "Understand text preprocessing, embeddings, NLP pipelines, and language model fundamentals.",
            "skills": "nlp,python,machine learning,text processing,embeddings",
            "level": "Intermediate"
        },
        {
            "title": "Data Visualization with Tableau",
            "description": "Create dashboards and visual reports to communicate insights effectively.",
            "skills": "tableau,data visualization,dashboards,analytics",
            "level": "Beginner"
        },
        {
            "title": "Statistics for Machine Learning",
            "description": "Build strong foundations in probability, distributions, and statistics for machine learning.",
            "skills": "statistics,probability,machine learning,math",
            "level": "Intermediate"
        },
        {
            "title": "Power BI for Business Intelligence",
            "description": "Build interactive dashboards and business reports using Power BI.",
            "skills": "power bi,business intelligence,data visualization,reporting",
            "level": "Beginner"
        },
        {
            "title": "Computer Vision Essentials",
            "description": "Learn image classification, CNNs, and computer vision fundamentals using deep learning.",
            "skills": "computer vision,deep learning,python,cnn,ai",
            "level": "Advanced"
        },
        {
            "title": "Product Management for AI",
            "description": "Learn AI product strategy, roadmap thinking, metrics, experimentation, stakeholder alignment, and tradeoffs.",
            "skills": "product management,ai,metrics,strategy,roadmap,experimentation",
            "level": "Intermediate"
        },
        {
            "title": "Generative AI for Product Managers",
            "description": "Understand LLMs, prompt design, evaluation, AI risks, and how to build generative AI products.",
            "skills": "generative ai,llms,prompting,ai product management,evaluation",
            "level": "Intermediate"
        },
        {
            "title": "User Research and Product Discovery",
            "description": "Learn customer interviews, problem discovery, user journeys, and product opportunity identification.",
            "skills": "user research,product discovery,customer interviews,ux,product management",
            "level": "Beginner"
        }
    ]
    return pd.DataFrame(data)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_user_profile(interests: str, skills: str, goal: str) -> str:
    parts = [interests.strip(), skills.strip(), goal.strip()]
    return " ".join([p for p in parts if p])

def prepare_course_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["title"].fillna("") + ". " +
        df["description"].fillna("") + ". " +
        df["skills"].fillna("") + ". " +
        df["level"].fillna("")
    )

def generate_recommendations(user_profile: str, df: pd.DataFrame, model, top_k: int = 5) -> pd.DataFrame:
    course_text = prepare_course_text(df).tolist()

    course_embeddings = model.encode(course_text)
    user_embedding = model.encode([user_profile])

    similarity_scores = cosine_similarity(user_embedding, course_embeddings)[0]

    results = df.copy()
    results["score"] = similarity_scores
    results = results.sort_values(by="score", ascending=False).head(top_k)

    return results

def get_reason(course_row: pd.Series, user_profile: str) -> str:
    user_words = set(user_profile.lower().replace(",", " ").split())
    course_words = set((str(course_row["skills"]) + " " + str(course_row["description"])).lower().replace(",", " ").split())
    overlap = list(user_words.intersection(course_words))

    if overlap:
        return f"Matched on: {', '.join(overlap[:3])}"
    return "Recommended based on semantic similarity to your profile."

st.title("AI-Powered Course Recommendation System")
st.markdown("Get personalized course recommendations based on your interests, skills, and career goals.")

df = load_data()
model = load_model()

with st.sidebar:
    st.header("Your Profile")
    interests = st.text_input("Interests", placeholder="e.g. product management, AI")
    skills = st.text_input("Existing Skills", placeholder="e.g. communication, analytics")
    goal = st.text_input("Career Goal", placeholder="e.g. become an AI Product Manager")
    top_k = st.slider("Number of Recommendations", min_value=3, max_value=8, value=5)

if st.button("Recommend Courses"):
    user_profile = build_user_profile(interests, skills, goal)

    if not user_profile.strip():
        st.warning("Please enter at least one of: interests, skills, or career goal.")
    else:
        recommendations = generate_recommendations(user_profile, df, model, top_k=top_k)

        st.subheader("Top Recommended Courses")

        for _, row in recommendations.iterrows():
            reason = get_reason(row, user_profile)

            with st.container():
                st.markdown(f"### {row['title']}")
                st.write(f"**Level:** {row['level']}")
                st.write(f"**Description:** {row['description']}")
                st.write(f"**Skills:** {row['skills']}")
                st.write(f"**Recommendation Score:** {row['score']:.3f}")
                st.write(f"**Why recommended:** {reason}")
                st.markdown("---")

st.subheader("Course Catalog")
st.dataframe(df, use_container_width=True)