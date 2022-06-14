import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

import pandas as pd
import numpy as np
import pickle
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

matplotlib.use('Agg')

# Features names
feature_names_best = ['Q3A', 'Q5A', 'Q10A', 'Q13A', 'Q16A', 'Q17A', 'Q21A', 'Q24A', 'Q26A', 'Q31A', 'Q34A', 'Q37A',
                      'Q38A', 'Q42A', 'gender', 'age', 'married']

gender_dict = {"Male": 1, "Female": 2, "Other": 3}
married_dict = {"Single": 1, "Married": 2, "Divorced": 3}
feature_dict = {"Did not apply to me at all": 0, "Applied to me to some degree, or some of the time": 1,
                "Applied to me to a considerable degree, or a good part of the time": 2,
                "Applied to me very much, or most of the time": 3}


# Function for Prediction Page
def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return key


def get_fvalue(val):
    feature_dict = {"Did not apply to me at all": 0, "Applied to me to some degree, or some of the time": 1,
                    "Applied to me to a considerable degree, or a good part of the time": 2,
                    "Applied to me very much, or most of the time": 3}
    for key, value in feature_dict.items():
        if val == key:
            return value


def load_model():
    loaded_model = pickle.load(open("depression_model.sav", 'rb'))
    return loaded_model


def norm(x):
    mean = [[23.448104, 1.793421, 1.176493, 1.231427, 1.517369, 1.448104, 1.785837, 1.517704, 1.657113, 1.352357,
             1.442196, 1.661638, 1.372428, 1.630086, 1.372847, 1.392751, 1.676556]]
    std = [[8.648522, 0.440547, 0.458494, 1.040715, 1.069513, 1.142576, 1.075868, 1.112149, 1.158984, 1.167680,
            1.052560, 1.068762, 1.043430, 1.151607, 1.141410, 1.188625, 1.031444]]
    return (x - mean) / std


def get_data():
    playground_data = pd.read_csv("clean_data.csv")
    playground_data.drop("Age_Groups", inplace=True, axis=1)
    playground_data.drop("Total_Count", inplace=True, axis=1)
    playground_data['age'] = playground_data['age'].replace([1996, 1998, 1993, 223, 1991], [23, 21, 26, 23, 28])

    playground_data = playground_data.replace("Normal", 1)
    playground_data = playground_data.replace("Mild", 2)
    playground_data = playground_data.replace("Moderate", 3)
    playground_data = playground_data.replace("Severe", 4)
    playground_data = playground_data.replace("Extremely Severe", 5)

    return playground_data


# Function for Dataset Page
def explore(data):
    data.drop("Condition", inplace=True, axis=1)
    df_types = pd.DataFrame(data.dtypes, columns=['Data Type'])
    numerical_cols = df_types[~df_types['Data Type'].isin(['object',
                                                           'bool'])].index.values
    df_types['Count'] = data.count()
    df_types['Unique Values'] = data.nunique()
    df_types['Min'] = data[numerical_cols].min()
    df_types['Max'] = data[numerical_cols].max()
    df_types['Average'] = data[numerical_cols].mean()
    df_types['Median'] = data[numerical_cols].median()
    df_types['St. Dev.'] = data[numerical_cols].std()
    return df_types.astype(str)


def main():
    page_icon = Image.open("Assets/depression_image.png")
    st.set_page_config(
        page_title="Depression Indicator",
        page_icon=page_icon,
    )

    hide_st_style = """
                    <style>
                    header {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    st.write('<style>div.block-container{padding-top:0rem;padding-bottom:0rem;}</style>', unsafe_allow_html=True)

    with st.sidebar:
        st.title("Depression Indicator")
        selected = option_menu(
            menu_title=None,
            options=["Home", "Prediction", "Treatment", "Dataset", "Model Information", "About Us"],
            icons=["house", "clipboard-check", "journal-medical", "table", "wrench", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Home":
        st.title("Home")
        st.text("")
        mainPage()

    elif selected == "Prediction":
        st.title("Prediction")
        st.text("")
        predictionResult()

    elif selected == "Treatment":
        st.title("Treatment")
        st.text("")
        treatmentPage()

    elif selected == "Dataset":
        st.title("Dataset")
        st.text("")
        datasetPage()

    elif selected == "Model Information":
        st.title("Model Information")
        st.text("")
        model_information()

    elif selected == "About Us":
        st.title("About Us")
        st.text("")
        aboutUsPage()


# Function for Dataset Page
def explore(data):
    df_types = pd.DataFrame(data.dtypes, columns=['Data Type'])
    numerical_cols = df_types[~df_types['Data Type'].isin(['object',
                                                           'bool'])].index.values
    df_types['Count'] = data.count()
    df_types['Unique Values'] = data.nunique()
    df_types['Min'] = data[numerical_cols].min()
    df_types['Max'] = data[numerical_cols].max()
    df_types['Average'] = data[numerical_cols].mean()
    df_types['Median'] = data[numerical_cols].median()
    df_types['St. Dev.'] = data[numerical_cols].std()
    return df_types.astype(str)


def mainPage():
    st.image("Assets/depression.jpg")
    st.text("")

    st.subheader("What is Depression")
    st.text("")
    st.markdown("""  
                Depression is a common mental disorder. Globally, it is estimated that 5% of adults suffer from the disorder. It is characterized by persistent sadness and a lack of interest or pleasure in previously rewarding or enjoyable activities. It can also disturb sleep and appetite. Tiredness and poor concentration are common. Depression is a leading cause of disability around the world and contributes greatly to the global burden of disease. The effects of depression can be long-lasting or recurrent and can dramatically affect a person’s ability to function and live a rewarding life.
                """)
    st.text(" ")

    st.subheader("How Depression is Classified")
    st.text("")
    st.markdown("""
                It’s common to feel down from time to time, but depression is a separate condition that should be treated with care. Aside from causing a general feeling of sadness, depression is known for causing feelings of hopelessness that don’t seem to go away.

                The term “depression” has become common in mainstream society. But depression is a more nuanced subject than popular usage may suggest. For one, not all cases of depression are the same. There are varying classifications of depression, and each can affect your life in different ways.

                Depression may be classified as:
                """)

    st.write("• Mild ")
    st.write("• Moderate ")
    st.write("• Severe, also called “Major” ")

    st.markdown("""
                The exact classification is based on many factors. These include the types of symptoms you experience, their severity, and how often they occur. Certain types of depression can also cause a temporary spike in the severity of symptoms.

                To learn more about your current state of depression, let's have a look at Prediction page.
                """)


def treatmentPage():
    submenu = ["Mild Depression", "Moderate Depression", "Severe Depression"]

    st.write("Different types of depression have their own symptoms and treatment. Please check the symptoms and "
             "suggestions that suit with your depression level.")
    st.text("")

    st.subheader("Type of Depression")
    activity = st.selectbox("Please select one type of depression", submenu)

    if activity == "Mild Depression":
        mild_depression()
    elif activity == "Moderate Depression":
        moderate_depression()
    elif activity == "Severe Depression":
        severe_depression()


def mild_depression():
    st.header('Mild Depression')
    st.subheader("What does mild depression feel like?")
    st.markdown("""
    Mild depression involves more than just feeling blue temporarily. Your symptoms can go on for days and are noticeable enough to interfere with your usual activities.
    """)

    st.text(" ")

    st.subheader("Symptoms of Mild Depression")
    st.markdown("""
    * Irritability or anger

    * Hopelessness

    * Feelings of guilt and despair

    * Self-loathing

    * A loss of interest in activities you once enjoyed

    * Difficulties concentrating at work

    * A lack of motivation

    * A sudden disinterest in socializing

    * Aches and pains with seemingly no direct cause

    * Daytime sleepiness and fatigue

    * Insomnia

    * Appetite changes

    * Weight changes

    * Reckless behavior, such as abuse of alcohol and drugs, or gambling

    If your symptoms persist for most of the day, on an average of four days a week for two years, you would most likely be diagnosed with persistent depressive disorder. This condition is also referred to as dysthymia.

    Though mild depression is noticeable, it’s the most difficult to diagnose. It’s easy to dismiss the symptoms and avoid discussing them with your doctor.
    """)

    st.text(" ")

    st.subheader("Suggestions to treat Mild Depression")
    st.markdown("""
    Despite the challenges in diagnosis, mild depression is the easiest to treat. Certain lifestyle changes can go a long way in boosting serotonin levels in the brain, which can help fight depressive symptoms.

    Helpful lifestyle changes include:

    * Exercising daily

    * Adhering to a sleep schedule

    * Eating a balanced diet rich in fruits and vegetables

    * Practicing yoga or meditation

    * Doing activities that reduce stress, such as journaling, reading, or listening to music


    Other treatments for mild depression include alternative remedies, such as St. John’s Wort and melatonin supplements. However, supplements can interfere with certain medications. Be sure to ask your doctor before taking any supplements for depression.

    A class of antidepressants called selective serotonin reuptake inhibitors (SSRIs) may be used in some cases. However, these tend to be more effective in people with more severe forms of depression. Recurrent depression tends to respond better to lifestyle changes and forms of talk therapy, such as psychotherapy, than medication.

    While medical treatment may not be needed, mild depression won’t necessarily go away on its own. In fact, when left alone, mild depression can progress to more severe forms.
    """)

    st.text(" ")


def moderate_depression():
    st.header('Moderate Depression')
    st.subheader("What does moderate depression feel like?")
    st.markdown("""
    In terms of symptomatic severity, moderate depression is the next level up from mild cases. Moderate and mild depression share similar symptoms.
    """)

    st.text(" ")

    st.subheader("Symptoms of Moderate Depression")
    st.markdown("""
    * Irritability or anger

    * Hopelessness

    * Feelings of guilt and despair

    * Self-loathing

    * A loss of interest in activities you once enjoyed

    * Difficulties concentrating at work

    * A lack of motivation

    * A sudden disinterest in socializing

    * Aches and pains with seemingly no direct cause

    * Daytime sleepiness and fatigue

    * Insomnia

    * Appetite changes

    * Weight changes

    * Reckless behavior, such as abuse of alcohol and drugs, or gambling

    * Problems with self-esteem

    * Reduced productivity

    * Feelings of worthlessness

    * Increased sensitivities

    * Excessive worrying

    The greatest difference is that the symptoms of moderate depression are severe enough to cause problems at home and work. You may also find significant difficulties in your social life.
    """)

    st.text(" ")

    st.subheader("Suggestions to treat Moderate Depression")
    st.markdown("""
    Moderate depression is easier to diagnose than mild cases because the symptoms significantly impact your daily life. The key to a diagnosis, though, is to make sure you talk to your doctor about the symptoms you’re experiencing.

    SSRIs, such as sertraline (Zoloft) or paroxetine (Paxil), may be prescribed. These medications can take up to six weeks to take full effect. Cognitive behavioral therapy (CBT) is also used in some cases of moderate depression.
    """)

    st.text(" ")


def severe_depression():
    st.header("Severe Depression")
    st.subheader("What does severe (major) depression feel like?")
    st.markdown("""
    Severe (major) depression is classified as having the symptoms of mild to moderate depression, but the symptoms are severe and noticeable, even to your loved ones.

    Episodes of major depression last an average of six months or longer. Sometimes severe depression can go away after a while, but it can also be recurrent for some people.

    Diagnosis is especially crucial in severe depression, and it may even be time-sensitive.
    """)

    st.text(" ")

    st.subheader("Symptoms of Severe Depression")
    st.markdown("""
    * Irritability or anger

    * Hopelessness

    * Feelings of guilt and despair

    * Self-loathing

    * A loss of interest in activities you once enjoyed

    * Difficulties concentrating at work

    * A lack of motivation

    * A sudden disinterest in socializing
    
    * Aches and pains with seemingly no direct cause

    * Daytime sleepiness and fatigue

    * Insomnia

    * Appetite changes

    * Weight changes

    * Reckless behavior, such as abuse of alcohol and drugs, or gambling

    * Problems with self-esteem

    * Reduced productivity

    * Feelings of worthlessness

    * Increased sensitivities

    * Excessive worrying

    * Delusions

    * Feelings of stupor

    * Hallucinations

    * Suicidal thoughts or behaviors

    """)

    st.text(" ")

    st.subheader("Suggestions to treat Severe Depression")
    st.markdown("""
    Severe depression requires medical treatment as soon as possible. Your doctor will likely recommend an SSRI and some form of talk therapy.

    If you’re experiencing suicidal thoughts or behaviors, you should seek immediate medical attention. Call your local emergency services or the National Suicide Prevention Lifeline at 800-273-8255 right away.
    """)

    st.text(" ")


def predictionResult():
    st.subheader("Survey")
    st.write("Please fill all of the questions below so that our model can predict your depression level.")
    st.text("")

    age = st.number_input("Age", 10, 120)
    gender = st.radio("Gender", tuple(gender_dict.keys()))
    married = st.radio("Marital Status", tuple(married_dict.keys()))
    Q3A = st.radio(" 1.  I couldn't seem to experience any positive feeling at all.",
                   tuple(feature_dict.keys()))
    Q5A = st.radio(" 2.  I just couldn't seem to get going.", tuple(feature_dict.keys()))
    Q10A = st.radio(" 3.  I felt that I had nothing to look forward to.", tuple(feature_dict.keys()))
    Q13A = st.radio(" 4.  I felt sad and depressed.", tuple(feature_dict.keys()))
    Q16A = st.radio(" 5.  I felt that I had lost interest in just about everything.",
                    tuple(feature_dict.keys()))
    Q17A = st.radio(" 6.  I felt I wasn't worth much as a person.", tuple(feature_dict.keys()))
    Q21A = st.radio(" 7.  I felt that life wasn't worthwhile.", tuple(feature_dict.keys()))
    Q24A = st.radio(" 8.  I couldn't seem to get any enjoyment out of the things I did.",
                    tuple(feature_dict.keys()))
    Q26A = st.radio(" 9.  I felt down-hearted and blue.", tuple(feature_dict.keys()))
    Q31A = st.radio("10.  I was unable to become enthusiastic about anything.", tuple(feature_dict.keys()))
    Q34A = st.radio("11.  I felt I was pretty worthless.", tuple(feature_dict.keys()))
    Q37A = st.radio("12.  I could see nothing in the future to be hopeful about.",
                    tuple(feature_dict.keys()))
    Q38A = st.radio("13.  I felt that life was meaningless.", tuple(feature_dict.keys()))
    Q42A = st.radio("14.  I found it difficult to work up the initiative to do things.",
                    tuple(feature_dict.keys()))
    feature_list = [age, get_value(gender, gender_dict), get_value(married, married_dict),
                    get_fvalue(Q3A), get_fvalue(Q5A), get_fvalue(Q10A), get_fvalue(Q13A),
                    get_fvalue(Q16A), get_fvalue(Q17A), get_fvalue(Q21A), get_fvalue(Q24A),
                    get_fvalue(Q26A), get_fvalue(Q31A), get_fvalue(Q34A), get_fvalue(Q37A),
                    get_fvalue(Q38A), get_fvalue(Q42A)]

    st.text("")
    st.subheader("Your Selection")
    st.write("Verify that all of your options chosen are as below.")
    st.text("")
    st.write(len(feature_list))
    pretty_result = {"Age": age, "Gender": gender, "Married": married, "Q1": Q3A, "Q2": Q5A,
                     "Q3": Q10A, "Q4": Q13A, "Q5": Q16A, "Q6": Q17A, "Q7": Q21A, "Q8": Q24A,
                     "Q9": Q26A, "Q10": Q31A, "Q11": Q34A, "Q12": Q37A, "Q13": Q38A,
                     "Q14": Q42A}
    st.json(pretty_result)
    user_input = np.array(feature_list).reshape(1, -1)

    st.text("")
    st.write("Click the Predict button to predict your depression level.")

    if st.button("Predict"):
        st.text("")
        st.subheader("Result")
        loaded_model = load_model()
        norm_user_input = norm(user_input)

        prediction = loaded_model.predict(norm_user_input)
        prediction_result = ""
        st.write("Prediction")
        if prediction == [[1]]:
            prediction_result = "Normal"
        elif prediction == [[2]]:
            prediction_result = "Mild Depression"
        elif prediction == [[3]]:
            prediction_result = "Moderate Depression"
        elif prediction == [[4]]:
            prediction_result = "Severe Depression"
        elif prediction == [[5]]:
            prediction_result = "Extremely Severe"
        st.code(prediction_result)

        st.text("")

        if prediction_result == "Normal":
            st.write("You did a great job! Keep motivate and don't let depression to beat your life!")

        elif prediction_result == "Mild Depression":
            st.write("Sometimes it's ok to have some minor depression in your life.")
            st.write("Check out the Treatment page to help yourself for treating your depression.")

        elif prediction_result == "Moderate Depression":
            st.write("Even though your depression level isn't quite serious, but taking an initiative to treat it "
                     "is better than nothing right?")
            st.write("Check out the Treatment page to help yourself for treating your depression.")

        elif prediction_result == "Severe Depression":
            st.write("It's seemed that you have a serious problem with depression. In order to help yourself, "
                     "please consult with doctors.")
            st.write("Check out the Treatment page to help yourself for treating your depression.")

        elif prediction_result == "Extremely Severe":
            st.write("I know that our life is hard, full of everything that isn't in our control. But please "
                     "appreciate your life, try to learn on how to love yourself and seek treatment from doctors. "
                     "Don't forget that our life also has lots of happy little moments, you just need to wait for "
                     "the right time for them to appear.")
            st.write("Check out the Treatment page to help yourself for treating your depression.")


def model_information():
    st.write("The prediction model that we use in our project is Support Vector Machine (SVM) with radial basis "
             "function as the kernel. Starting from data preparation, to data processing and finally to model "
             "preparation, we have utilised pandas, numpy and scikit-learn libraries.")
    st.text("")

    st.subheader("People Condition of Depression Level")
    st.write("Here is the bar chart graph that shows the total labels count for our dataset")
    st.text("")
    label_data = pd.read_csv("clean_data.csv")
    label_data['Condition'] = label_data['Condition'].replace([1, 2, 3, 4, 5],
                                              ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe'])
    label_data['Condition'].value_counts().plot(kind='bar')
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.text("")

    st.subheader("Correlation of Depression")
    st.write("Correlation graph allows us to see how well the relationship between one variable with another. The "
             "lighter the tile colour, the stronger the correlation between two variables.")
    st.text("")
    corr_data = get_data().copy()
    fig, ax = plt.subplots()
    sns.heatmap(corr_data.corr(), ax=ax)
    st.write(fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.text("")

    st.subheader("Learning Curve")
    st.write("Learning curve able to show us how the performance of the classifier changes. At here, we are using "
             "Support Vector Machine (SVM) with the RBF Kernel.")
    st.text("")
    st.image("Assets/learning_curve.png")
    st.text("")

    st.subheader("Performance Against Unseen Data")
    st.write("By utilising the training data and k-fold cross-validation, we could shuffle and fold the data to "
             "evaluate machine learning model when facing unseen data. That is, to use a limited sample in order to "
             "estimate how the model is expected to perform in general when used to make predictions on data not used "
             "during the training of the model.")
    st.text("")
    st.image("Assets/performance_against_unseen_data.png")
    st.text("")

    st.subheader("Confusion Matrix")
    st.write("Confusion Matrix is a performance measurement for machine learning classification where output can be "
             "two or more classes.")
    st.write("The table compares both true labels and the predicted labels by the model. The value on the diagonal "
             "refer to the number of values that are being correctly predicted by the model, whereas the rest are the "
             "number of values that are predicted wrongly.")
    st.image("Assets/confusion_matrix.png")
    st.text("")

    st.subheader("Decision Boundary")
    st.write("SVM will try to make a decision boundary in such as a way that the separation betweeen the classes is as "
             "wide as possible.")
    st.write("Since our model has high dimension, we will have to reduce its dimensionality and use linear SVM to plot "
             "to plot the boundary.")
    st.image("Assets/decision_boundary.png")
    st.text("")


def datasetPage():
    st.write("Sometimes we are wondered how's the dataset looks like. Well, here's the great news for you! At here, we "
             "provide some functionality for the user to peek through the data easily and display information of them.")
    st.text("")
    inspect_data = get_data()

    st.subheader("About Dataset")
    st.write("The dataset that we use is obtained from this [link](https://www.kaggle.com/datasets/lucasgreenwell/depression-anxiety-stress-scales-responses?resource=download&select=data.csv).")
    st.write("It consists of questions, answers and metadata collected from 39775 Depression Anxiety Stress Scales. "  
             "The data was hosted on OpenPsychometrics.org. We utilize this data to train the machine learning model.")
    st.text("")

    st.subheader("Full Dataset")
    st.write("Below are the whole dataset that has been used by our model")
    st.dataframe(inspect_data)
    st.text("")
    st.write("Some additional functions for those endeavour")

    if st.checkbox("Show Dataset for a range of rows"):
        number = st.number_input("Number of rows to view", 10)
        st.dataframe(inspect_data.head(number))

    if st.checkbox("Column Names"):
        st.write(inspect_data.columns)

    if st.checkbox("Shape of Dataset"):
        st.text("Dimension in (Row, Column)")
        st.write(inspect_data.shape)

    if st.checkbox("Check Specific Columns"):
        all_columns = inspect_data.columns.tolist()
        selected_columns = st.multiselect("Select columns", all_columns)
        new_dataframe = inspect_data[selected_columns]
        st.dataframe(new_dataframe)

    if st.checkbox("Show value counts"):
        st.text("Value counts by target/class")
        st.text("1 - Normal, 2 - Mild, 3 - Moderate, 4 - Severe, 5 - Extremely Severe")
        st.write(inspect_data.iloc[:, -1].value_counts())

    if st.checkbox("Show data types"):
        dataset = get_data()
        st.write(explore(dataset))

    if st.checkbox("Show summary"):
        st.write(inspect_data.describe().T)

    st.text("")

    st.subheader("Value Counts in each Column")
    st.write("In order to know how many value counts for each column, here is the bar chart tool that provides such "
             "functionality.")
    st.text("")
    df = pd.read_csv("clean_data.csv")
    submenu = ["Optimistic", "Motivation", "Looking-Forward", "Sadness", "Interest", "Existential-Crisis",
               "Importance", "Enjoyment", "Down-hearted", "Enthusiasm", "Worthiness", "Hopefulness", "Meaningless",
               "Tiredness", "Condition"]
    data = st.selectbox("Please select a column", submenu)
    if data == "Optimistic":
        st.subheader("People who are optimistic")
        df['Optimistic'] = df['Optimistic'].replace([0, 1, 2, 3],
                                                    ['Did not applied to me at all',
                                                     'Applied to me to some degree, or some of the time',
                                                     'Applied to me to a considerable degree, or a good part of the time',
                                                     'Applied to me very much, or most of the time'])
        df['Optimistic'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Motivation":
        st.subheader("People who are motivated")
        df['Motivation'] = df['Motivation'].replace([0, 1, 2, 3],
                                                    ['Did not applied to me at all',
                                                     'Applied to me to some degree, or some of the time',
                                                     'Applied to me to a considerable degree, or a good part of the time',
                                                     'Applied to me very much, or most of the time'])
        df['Motivation'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Looking-Forward":
        st.subheader("People who are looking-forward")
        df['Looking-Forward'] = df['Looking-Forward'].replace([0, 1, 2, 3],
                                                              ['Did not applied to me at all',
                                                               'Applied to me to some degree, or some of the time',
                                                               'Applied to me to a considerable degree, or a good part of the time',
                                                               'Applied to me very much, or most of the time'])
        df['Looking-Forward'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Sadness":
        st.subheader("People who are always feel sad and depressed")
        df['Sadness'] = df['Sadness'].replace([0, 1, 2, 3],
                                              ['Did not applied to me at all',
                                               'Applied to me to some degree, or some of the time',
                                               'Applied to me to a considerable degree, or a good part of the time',
                                               'Applied to me very much, or most of the time'])
        df['Sadness'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Interest":
        st.subheader("People who are lost interest of everything")
        df['Interest'] = df['Interest'].replace([0, 1, 2, 3],
                                                ['Did not applied to me at all',
                                                 'Applied to me to some degree, or some of the time',
                                                 'Applied to me to a considerable degree, or a good part of the time',
                                                 'Applied to me very much, or most of the time'])
        df['Interest'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Existential-Crisis":
        st.subheader("People who are doubt on their existence")
        df['Existential-Crisis'] = df['Existential-Crisis'].replace([0, 1, 2, 3],
                                                                    ['Did not applied to me at all',
                                                                     'Applied to me to some degree, or some of the time',
                                                                     'Applied to me to a considerable degree, or a good part of the time',
                                                                     'Applied to me very much, or most of the time'])
        df['Existential-Crisis'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Importance":
        st.subheader("People who does not have the feeling of worthwhile")
        df['Importance'] = df['Importance'].replace([0, 1, 2, 3],
                                                    ['Did not applied to me at all',
                                                     'Applied to me to some degree, or some of the time',
                                                     'Applied to me to a considerable degree, or a good part of the time',
                                                     'Applied to me very much, or most of the time'])
        df['Importance'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Enjoyment":
        st.subheader("People who does not enjoy on anything they have done")
        df['Enjoyment'] = df['Enjoyment'].replace([0, 1, 2, 3],
                                                  ['Did not applied to me at all',
                                                   'Applied to me to some degree, or some of the time',
                                                   'Applied to me to a considerable degree, or a good part of the time',
                                                   'Applied to me very much, or most of the time'])
        df['Enjoyment'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Down-hearted":
        st.subheader("People who feels discouraged and emotionally down")
        df['Down-hearted'] = df['Down-hearted'].replace([0, 1, 2, 3],
                                                        ['Did not applied to me at all',
                                                         'Applied to me to some degree, or some of the time',
                                                         'Applied to me to a considerable degree, or a good part of the time',
                                                         'Applied to me very much, or most of the time'])
        df['Down-hearted'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Enthusiasm":
        st.subheader("People who does not have enthusiastic on anything")
        df['Enthusiasm'] = df['Enthusiasm'].replace([0, 1, 2, 3],
                                                    ['Did not applied to me at all',
                                                     'Applied to me to some degree, or some of the time',
                                                     'Applied to me to a considerable degree, or a good part of the time',
                                                     'Applied to me very much, or most of the time'])
        df['Enthusiasm'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Worthiness":
        st.subheader("People who are doubt on their existence")
        df['Worthiness'] = df['Worthiness'].replace([0, 1, 2, 3],
                                                    ['Did not applied to me at all',
                                                     'Applied to me to some degree, or some of the time',
                                                     'Applied to me to a considerable degree, or a good part of the time',
                                                     'Applied to me very much, or most of the time'])
        df['Worthiness'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Hopefulness":
        st.subheader("People who does not have any hope in their future")
        df['Hopefulness'] = df['Hopefulness'].replace([0, 1, 2, 3],
                                                      ['Did not applied to me at all',
                                                       'Applied to me to some degree, or some of the time',
                                                       'Applied to me to a considerable degree, or a good part of the time',
                                                       'Applied to me very much, or most of the time'])
        df['Hopefulness'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Meaningless":
        st.subheader("People who feel their life is meaningless")
        df['Meaningless'] = df['Meaningless'].replace([0, 1, 2, 3],
                                                      ['Did not applied to me at all',
                                                       'Applied to me to some degree, or some of the time',
                                                       'Applied to me to a considerable degree, or a good part of the time',
                                                       'Applied to me very much, or most of the time'])
        df['Meaningless'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Tiredness":
        st.subheader("People who does not have the initiative to do things")
        df['Tiredness'] = df['Tiredness'].replace([0, 1, 2, 3],
                                                  ['Did not applied to me at all',
                                                   'Applied to me to some degree, or some of the time',
                                                   'Applied to me to a considerable degree, or a good part of the time',
                                                   'Applied to me very much, or most of the time'])
        df['Tiredness'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Condition":
        st.subheader("The depression level of the participants")
        df['Condition'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    st.text("")

    st.subheader("Data Visualization")
    st.write("People surely love graph more than boring text! At this section, you can customize your own plot.")
    st.text("")
    all_columns_names = inspect_data.columns.tolist()
    type_of_plot = st.selectbox("Select type of plot", ["Area", "Bar", "Line", "Histogram", "Box", "Pie Plot",
                                                        "Kernel Density Estimation"])
    selected_columns_names = st.multiselect("Select columns to plot", all_columns_names)

    if st.button("Generate Plot"):
        st.success("Generating customizable plot of {} for {}".format(type_of_plot, selected_columns_names))
        if type_of_plot == "Area":
            custom_data = inspect_data[selected_columns_names]
            st.area_chart(custom_data)

        elif type_of_plot == "Bar":
            custom_data = inspect_data[selected_columns_names]
            st.bar_chart(custom_data)

        elif type_of_plot == "Line":
            custom_data = inspect_data[selected_columns_names]
            st.line_chart(custom_data)

        elif type_of_plot == "Pie Plot":
            st.write(inspect_data.iloc[:, -1].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()

        elif type_of_plot == "Box":
            custom_plot = inspect_data[selected_columns_names].plot(kind="box")
            st.write(custom_plot)
            st.pyplot()

        elif type_of_plot == "Histogram":
            custom_plot = inspect_data[selected_columns_names].plot(kind="hist")
            st.write(custom_plot)
            st.pyplot()

        elif type_of_plot == "Kernel Density Estimation":
            custom_plot = inspect_data[selected_columns_names].plot(kind="kde")
            st.write(custom_plot)
            st.pyplot()


def aboutUsPage():
    st.subheader("Problem Statement")
    st.write("""Based on World Health Organization (WHO), Depression is a common illness worldwide, with an estimated 
                3.8% of the population affected, including 5.0% among adults and 5.7% among adults older than 60 years. 
                Depression is a leading cause of disability worldwide and is a major contributor to the overall global 
                burden of disease. More women are affected by depression than men. Depression can lead to suicide. 
                However, there is effective treatment for mild, moderate, and severe depression. """)

    st.text("")

    st.subheader("Objective")
    st.markdown("""
    * Raising awareness on depression in the community.
    * Estimate your potential of having depression.
    * Provide information about different types of depression, and suggestions to treat it.
    """)

    st.text("")

    st.subheader("Group Information")
    d = {"Name":["Jason Wong Jack","Wong Yan Jian","Chong Jia Ying","Lim Hon Ting","Lim JiaJun"],
         'Matric Number':["U2102864","U2102753","U2102853","S2114212","S2124035"]}
    df = pd.DataFrame(data=d, index=[1,2,3,4,5])
    st.table(df)
    st.text("")

    st.subheader("Project")
    st.write("The whole source code that has been used to make the prediction model and the website can be found on "
             "[GitHub](https://github.com/ryoshi007/DepressionIndicator). We hope that you can enjoy our hardwork. "
             "Thanks!")


if __name__ == '__main__':
    main()
