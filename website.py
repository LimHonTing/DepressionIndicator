import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

import pandas as pd
import numpy as np
import pickle
import csv

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

import sklearn as sk

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
    mean = [[23.390991, 1.796313, 1.173266, 1.217683, 1.515776, 1.436246, 1.781060, 1.512424, 1.653886, 1.342133,
             1.435240, 1.652629, 1.371423, 1.624597, 1.367484, 1.383742, 1.672240]]
    std = [[8.534579, 0.440800, 0.455465, 1.034644, 1.067342, 1.136569, 1.072174, 1.112710, 1.154055, 1.162131,
            1.051159, 1.065141, 1.043990, 1.147513, 1.138446, 1.185164, 1.031158]]
    return (x - mean) / std


def main():
    page_icon = Image.open("happy-icon-20.jpg")
    st.set_page_config(
        page_title="Depression Indicator",
        page_icon=page_icon,
    )

    hide_st_style = """
                    <style>
                    # header {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    st.write('<style>div.block-container{padding-top:0rem;padding-bottom:0rem;}</style>', unsafe_allow_html=True)

    with st.sidebar:
        st.title("Depression Indicator")
        selected = option_menu(
            menu_title=None,
            options=["Home", "Prediction", "Treatment", "Dataset", "Journey", "About Us"],
            icons=["house", "clipboard-check", "journal-medical", "table", "book", "info-circle"],
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
        predictionPage()

    elif selected == "Treatment":
        st.title("Treatment")
        st.text("")
        treatmentPage()
    elif selected == "Dataset":
        st.title("Dataset")
        st.text("")
        datasetPage()


def mainPage():
    st.image("depression.jpg")
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
    st.header('Mild Depression')
    st.subheader("What does mild depression feel like?")
    st.markdown("""
    Mild depression involves more than just feeling blue temporarily. Your symptoms can go on for days and are noticeable enough to interfere with your usual activities.
    """)

    st.text(" ")

    st.sidebar.markdown('''
            # Sections
            - [Mild Depression](#mild-depression)
            - [Moderate Depression](#moderate-depression)
            - [Severe Depression](#severe-depression)
            ''', unsafe_allow_html=False)

    st.subheader("Symptoms of Mild Depression")
    st.markdown("""
    •irritability or anger

    •hopelessness

    •feelings of guilt and despair

    •self-loathing

    •a loss of interest in activities you once enjoyed

    •difficulties concentrating at work

    •a lack of motivation

    •a sudden disinterest in socializing

    •aches and pains with seemingly no direct cause

    •daytime sleepiness and fatigue

    •insomnia

    •appetite changes

    •weight changes

    •reckless behavior, such as abuse of alcohol and drugs, or gambling

    If your symptoms persist for most of the day, on an average of four days a week for two years, you would most likely be diagnosed with persistent depressive disorder. This condition is also referred to as dysthymia.

    Though mild depression is noticeable, it’s the most difficult to diagnose. It’s easy to dismiss the symptoms and avoid discussing them with your doctor.
    """)

    st.text(" ")

    st.subheader("Suggestions to treat Mild Depression")
    st.markdown("""
    Despite the challenges in diagnosis, mild depression is the easiest to treat. Certain lifestyle changes can go a long way in boosting serotonin levels in the brain, which can help fight depressive symptoms.

    Helpful lifestyle changes include:

    •exercising daily

    •adhering to a sleep schedule

    •eating a balanced diet rich in fruits and vegetables

    •practicing yoga or meditation

    •doing activities that reduce stress, such as journaling, reading, or listening to music


    Other treatments for mild depression include alternative remedies, such as St. John’s Wort and melatonin supplements. However, supplements can interfere with certain medications. Be sure to ask your doctor before taking any supplements for depression.

    A class of antidepressants called selective serotonin reuptake inhibitors (SSRIs) may be used in some cases. However, these tend to be more effective in people with more severe forms of depression. Recurrent depression tends to respond better to lifestyle changes and forms of talk therapy, such as psychotherapy, than medication.

    While medical treatment may not be needed, mild depression won’t necessarily go away on its own. In fact, when left alone, mild depression can progress to more severe forms.
    """)

    st.text(" ")

    st.header('Moderate Depression')
    st.subheader("What does moderate depression feel like?")
    st.markdown("""
    In terms of symptomatic severity, moderate depression is the next level up from mild cases. Moderate and mild depression share similar symptoms.
    """)

    st.text(" ")

    st.subheader("Symptoms of Moderate Depression")
    st.markdown("""
    •irritability or anger

    •hopelessness

    •feelings of guilt and despair

    •self-loathing

    •a loss of interest in activities you once enjoyed

    •difficulties concentrating at work

    •a lack of motivation

    •a sudden disinterest in socializing

    •aches and pains with seemingly no direct cause

    •daytime sleepiness and fatigue

    •insomnia

    •appetite changes

    •weight changes

    •reckless behavior, such as abuse of alcohol and drugs, or gambling

    •problems with self-esteem

    •reduced productivity

    •feelings of worthlessness

    •increased sensitivities

    •excessive worrying

    The greatest difference is that the symptoms of moderate depression are severe enough to cause problems at home and work. You may also find significant difficulties in your social life.
    """)

    st.text(" ")

    st.subheader("Suggestions to treat Moderate Depression")
    st.markdown("""
    Moderate depression is easier to diagnose than mild cases because the symptoms significantly impact your daily life. The key to a diagnosis, though, is to make sure you talk to your doctor about the symptoms you’re experiencing.

    SSRIs, such as sertraline (Zoloft) or paroxetine (Paxil), may be prescribed. These medications can take up to six weeks to take full effect. Cognitive behavioral therapy (CBT) is also used in some cases of moderate depression.
    """)

    st.text(" ")

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
    •irritability or anger

    •hopelessness

    •feelings of guilt and despair

    •self-loathing

    •a loss of interest in activities you once enjoyed

    •difficulties concentrating at work

    •a lack of motivation

    •a sudden disinterest in socializing

    •aches and pains with seemingly no direct cause

    •daytime sleepiness and fatigue

    •insomnia

    •appetite changes

    •weight changes

    •reckless behavior, such as abuse of alcohol and drugs, or gambling

    •problems with self-esteem

    •reduced productivity

    •feelings of worthlessness

    •increased sensitivities

    •excessive worrying

    •delusions

    •feelings of stupor

    •hallucinations

    •suicidal thoughts or behaviors

    """)

    st.text(" ")

    st.subheader("Suggestions to treat Severe Depression")
    st.markdown("""
    Severe depression requires medical treatment as soon as possible. Your doctor will likely recommend an SSRI and some form of talk therapy.

    If you’re experiencing suicidal thoughts or behaviors, you should seek immediate medical attention. Call your local emergency services or the National Suicide Prevention Lifeline at 800-273-8255 right away.
    """)

    st.text(" ")


def predictionPage():
    submenu = ["Prediction", "Model Information", "Playground"]

    with st.sidebar:
        st.subheader("Activity")
        activity = st.selectbox("Please select one activity", submenu)

    if activity == "Model Information":
        # Condition count
        st.title("Data Vis Plot")
        df = pd.read_csv("clean_data.csv")
        st.dataframe(df)

        st.subheader("People Condition of Depression Level")
        df['Condition'].value_counts().plot(kind='bar')
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Depr.corr
        st.subheader("Correlation of Depression")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), ax=ax)
        st.write(fig)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        if st.checkbox("Area Chart"):
            all_columns = df.columns.to_list()
            feat_choices = st.multiselect("Choose a Feature", all_columns)
            new_df = df[feat_choices]
            st.area_chart(new_df)

    elif activity == "Prediction":
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


def datasetPage():
    st.header("Data Visualization")
    df = pd.read_csv("clean_data.csv")
    with open('clean_data.csv', newline='') as f:
        reader = csv.reader(f)
        submenu = ["Optimistic", "Motivation", "Looking-Forward", "Sadness", "Interest", "Existential-Crisis",
                   "Importance", "Enjoyment", "Down-hearted", "Enthusiasm", "Worthiness", "Hopefulness", "Meaningless",
                   "Tiredness", "Condition"]
    data = st.selectbox("Data", submenu)
    if data == "Optimistic":
        st.subheader("People who are optimistic")
        df['Optimistic'] = df['Optimistic'].replace([0, 1, 2, 3],
                                 ['Did not applied to me at all', 'Applied to me to some degree, or some of the time',
                                  'Applied to me to a considerable degree, or a good part of the time',
                                  'Applied to me very much, or most of the time'])
        df['Optimistic'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Motivation":
        st.subheader("People who are motivated")
        df['Motivation'] = df['Motivation'].replace([0, 1, 2, 3],
                                 ['Did not applied to me at all', 'Applied to me to some degree, or some of the time',
                                  'Applied to me to a considerable degree, or a good part of the time',
                                  'Applied to me very much, or most of the time'])
        df['Motivation'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Looking-Forward":
        st.subheader("People who are looking-forward")
        df['Looking-Forward'] = df['Looking-Forward'].replace([0, 1, 2, 3],
                                 ['Did not applied to me at all', 'Applied to me to some degree, or some of the time',
                                  'Applied to me to a considerable degree, or a good part of the time',
                                  'Applied to me very much, or most of the time'])
        df['Looking-Forward'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Sadness":
        st.subheader("People who are always feel sad and depressed")
        df['Sadness'] = df['Sadness'].replace([0, 1, 2, 3],
                                 ['Did not applied to me at all', 'Applied to me to some degree, or some of the time',
                                  'Applied to me to a considerable degree, or a good part of the time',
                                  'Applied to me very much, or most of the time'])
        df['Sadness'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Interest":
        st.subheader("People who are lost interest of everything")
        df['Interest'] = df['Interest'].replace([0, 1, 2, 3],
                                 ['Did not applied to me at all', 'Applied to me to some degree, or some of the time',
                                  'Applied to me to a considerable degree, or a good part of the time',
                                  'Applied to me very much, or most of the time'])
        df['Interest'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Existential-Crisis":
        st.subheader("People who are doubt on their existence")
        df['Existential-Crisis'] = df['Existential-Crisis'].replace([0, 1, 2, 3],
                                 ['Did not applied to me at all', 'Applied to me to some degree, or some of the time',
                                  'Applied to me to a considerable degree, or a good part of the time',
                                  'Applied to me very much, or most of the time'])
        df['Existential-Crisis'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Importance":
        st.subheader("People who does not have the feeling of worthwhile")
        df['Importance'] = df['Importance'].replace([0, 1, 2, 3],
                                 ['Did not applied to me at all', 'Applied to me to some degree, or some of the time',
                                  'Applied to me to a considerable degree, or a good part of the time',
                                  'Applied to me very much, or most of the time'])
        df['Importance'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Enjoyment":
        st.subheader("People who does not enjoy on anything they have done")
        df['Enjoyment'] = df['Enjoyment'].replace([0, 1, 2, 3],
                                 ['Did not applied to me at all', 'Applied to me to some degree, or some of the time',
                                  'Applied to me to a considerable degree, or a good part of the time',
                                  'Applied to me very much, or most of the time'])
        df['Enjoyment'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Down-hearted":
        st.subheader("People who feels discouraged and emotionally down")
        df['Down-hearted'] = df['Down-hearted'].replace([0, 1, 2, 3],
                                 ['Did not applied to me at all', 'Applied to me to some degree, or some of the time',
                                  'Applied to me to a considerable degree, or a good part of the time',
                                  'Applied to me very much, or most of the time'])
        df['Down-hearted'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Enthusiasm":
        st.subheader("People who does not have enthusiastic on anything")
        df['Enthusiasm'] = df['Enthusiasm'].replace([0, 1, 2, 3],
                                 ['Did not applied to me at all', 'Applied to me to some degree, or some of the time',
                                  'Applied to me to a considerable degree, or a good part of the time',
                                  'Applied to me very much, or most of the time'])
        df['Enthusiasm'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Worthiness":
        st.subheader("People who are doubt on their existence")
        df['Worthiness'] = df['Worthiness'].replace([0, 1, 2, 3],
                                 ['Did not applied to me at all', 'Applied to me to some degree, or some of the time',
                                  'Applied to me to a considerable degree, or a good part of the time',
                                  'Applied to me very much, or most of the time'])
        df['Worthiness'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Hopefulness":
        st.subheader("People who does not have any hope in their future")
        df['Hopefulness'] = df['Hopefulness'].replace([0, 1, 2, 3],
                                 ['Did not applied to me at all', 'Applied to me to some degree, or some of the time',
                                  'Applied to me to a considerable degree, or a good part of the time',
                                  'Applied to me very much, or most of the time'])
        df['Hopefulness'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Meaningless":
        st.subheader("People who feel their life is meaningless")
        df['Meaningless'] = df['Meaningless'].replace([0, 1, 2, 3],
                                 ['Did not applied to me at all', 'Applied to me to some degree, or some of the time',
                                  'Applied to me to a considerable degree, or a good part of the time',
                                  'Applied to me very much, or most of the time'])
        df['Meaningless'].value_counts().plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif data == "Tiredness":
        st.subheader("People who does not have the initiative to do things")
        df['Tiredness'] = df['Tiredness'].replace([0, 1, 2, 3],
                                 ['Did not applied to me at all', 'Applied to me to some degree, or some of the time',
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


if __name__ == '__main__':
    main()
