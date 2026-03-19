# AI-Powered Course Recommendation System

An AI-assisted course recommendation system that suggests relevant learning paths based on a user’s interests, existing skills, and career goals.

This project was built as an AI Product Management portfolio project to demonstrate how recommendation systems can improve course discovery, personalize learning experiences, and support better user outcomes.

---

## Problem Statement

Online learning platforms offer thousands of courses, but many users struggle to decide what to learn next. Generic recommendations often fail to reflect a learner’s real goals, current skill level, or intended career path.

This leads to:
- poor course discovery
- low engagement
- irrelevant enrollments
- lower completion rates

The core problem is helping users discover the right courses at the right time.

---

## Solution

This system uses a hybrid recommendation approach to generate personalized course suggestions.

It combines:
- semantic similarity using sentence embeddings
- role-based skill mapping
- category matching
- level preference matching

The application takes user inputs such as:
- interests
- existing skills
- career goal

It then recommends the most relevant courses and explains why they were selected.

---

## Key Features

- Personalized recommendations based on user profile
- Multi-role support for roles such as:
  - ML Engineer
  - Data Analyst
  - AI Product Manager
  - Backend Engineer
  - Frontend Engineer
  - Cloud Engineer
  - DevOps Engineer
  - UI/UX Designer
  - Cybersecurity Analyst
- Hybrid scoring using semantic similarity and structured product logic
- Recommendation reason generation
- Interactive Streamlit interface
- Course catalog view

---

## Product Thinking Behind the Project

This project was designed as more than a coding demo. It also reflects product management thinking around personalization systems.

Key product considerations included:
- **Cold start problem:** users may have no prior learning history, so the app uses onboarding-style inputs instead
- **Exploration vs exploitation:** recommendations should balance user intent with adjacent relevant learning opportunities
- **Career-goal alignment:** recommendations should reflect the user’s target role, not only broad keyword similarity
- **User trust:** the app explains why a course was recommended to improve transparency

---

## How It Works

1. The user enters interests, existing skills, and a career goal
2. The system builds a user profile from these inputs
3. Sentence embeddings are used to measure semantic similarity between the user profile and course descriptions
4. Additional ranking signals are applied:
   - role-to-skill match
   - category alignment
   - level preference
5. The system returns top-ranked courses with recommendation scores and reasons

---

## Tech Stack

- Python
- Streamlit
- Pandas
- Sentence Transformers
- Scikit-learn
- Torch

---

## Example Roles Supported

The system can generate recommendations for roles such as:
- Machine Learning Engineer
- Data Scientist
- Data Analyst
- Business Analyst
- AI Product Manager
- Product Manager
- Backend Engineer
- Frontend Engineer
- UI/UX Designer
- Cloud Engineer
- DevOps Engineer
- Cybersecurity Analyst

---

## Example Recommendation Logic

For a user with a goal such as **ML Engineer**, the system prioritizes courses related to:
- machine learning
- statistics
- deep learning
- MLOps
- feature engineering

For a user targeting **Data Analyst**, it prioritizes:
- SQL
- Excel
- Power BI
- Tableau
- dashboards and reporting

For **AI Product Manager**, it prioritizes:
- AI product strategy
- metrics
- experimentation
- user research
- generative AI fundamentals

---

## Project Structure

```bash
ai-course-recommender/
├── app.py
├── requirements.txt
└── README.md
