# revised code for stream lit
# packages
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# read in data
s = pd.read_csv("social_media_usage.csv")
# clean function
def clean_sm(x):
    return np.where(x == 1, 1, 0)
# create ss data frame
ss = s.copy()
ss['sm_li'] = clean_sm(ss['web1h'])
features = ['income', 'educ2', 'par', 'marital', 'gender', 'age']
ss = ss[['sm_li'] + features]
# handle missing data
ss['income'] = np.where(ss['income'] > 9, np.nan, ss['income'])
ss['educ2'] = np.where(ss['educ2'] > 8, np.nan, ss['educ2'])
ss['age'] = np.where(ss['age'] > 98, np.nan, ss['age'])
ss = ss.dropna()
# target vector and feature set
y = ss["sm_li"]
X = ss[features]

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.values, 
                                                    y,  
                                                    stratify = y,
                                                    test_size=0.2, 
                                                    random_state= 987)

# logistic regression model
lr = LogisticRegression(class_weight='balanced', random_state= 987)
lr.fit(X_train, y_train)

# predict function
def predict(lr, user_data):
    predicted_class = lr.predict([user_data])
    probs = lr.predict_proba([user_data])
    return predicted_class[0], probs[0][1]

# user input form in Streamlit
def user_input_form():
    # show descriptions for each feature on page
    st.write("### Income (household):")
    st.write("1. Less than $10,000")
    st.write("2. $10,000 - $20,000")
    st.write("3. $20,000 - $30,000")
    st.write("4. $30,000 - $40,000")
    st.write("5. $40,000 - $50,000")
    st.write("6. $50,000 - $75,000")
    st.write("7. $75,000 - $100,000")
    st.write("8. $100,000 - $150,000")
    st.write("9. $150,000 or more")
    
    income = st.selectbox("Select income level:", [1, 2, 3, 4, 5, 6, 7, 8, 9])

    st.write("### Education Level (highest school/degree completed):")
    st.write("1. Less than high school (Grades 1-8 or no formal schooling)")
    st.write("2. High school incomplete (Grades 9-11 or Grade 12 with NO diploma)")
    st.write("3. High school graduate (Grade 12 with diploma or GED certificate)")
    st.write("4. Some college, no degree (includes some community college)")
    st.write("5. Two-year associate degree from a college or university")
    st.write("6. Four-year college or university degree/Bachelor’s degree")
    st.write("7. Some postgraduate or professional schooling, no postgraduate degree")
    st.write("8. Postgraduate or professional degree (e.g., MA, MS, PhD, MD, JD)")
    
    educ2 = st.selectbox("Select education level:", [1, 2, 3, 4, 5, 6, 7, 8])

    st.write("### Are you a parent of a child under 18 living in your home?")
    st.write("1. Yes")
    st.write("2. No")
    
    par = st.radio("Select parent status:", [1, 2])

    st.write("### Marital Status:")
    st.write("1. Married")
    st.write("2. Living with a partner")
    st.write("3. Divorced")
    st.write("4. Separated")
    st.write("5. Widowed")
    st.write("6. Never been married")
    
    marital = st.selectbox("Select marital status:", [1, 2, 3, 4, 5, 6])

    st.write("### Gender:")
    st.write("1. Male")
    st.write("2. Female")
    st.write("3. Other")
    
    gender = st.selectbox("Select gender:", [1, 2, 3])

    st.write("### Age (numeric age):")
    age = st.slider("Select age:", min_value=18, max_value=100, value=30)

    # Collect user inputs into a list
    user_data = [income, educ2, par, marital, gender, age]
    return user_data

# add model to streamlit
def main():
    st.title("LinkedIn Usage Prediction")
    st.write("Enter the user's information to predict LinkedIn usage.")

    # Get user input data
    user_data = user_input_form()

    if st.button('Predict'):
        # Make prediction using the model
        predicted_class, prob = predict(lr, user_data)

        # Display prediction and probability
        if predicted_class == 1:
            st.write(f"The person is predicted to be a LinkedIn user.")
        else:
            st.write(f"The person is predicted to NOT be a LinkedIn user.")
        
        st.write(f"Probability of being a LinkedIn user: {prob:.2f}")

# Run the app
if __name__ == "__main__":
    main()
