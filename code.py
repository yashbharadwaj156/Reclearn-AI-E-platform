import streamlit as st
import streamlit.components.v1 as stc

#load EDA
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
# from fuzzywuzzy import process


#Load our Dataset:
def load_data(data):
    df = pd.read_csv(data)
    return df

#Vectorize
def vectorize_text_to_cosine_mat(data):
    count_vect= CountVectorizer()

#Cosine similarity Matrix
    cv_mat = count_vect.fit_transform(data)
    cosine_sim_mat=cosine_similarity(cv_mat)
    return cosine_sim_mat

#Course Recommendation System






#UDEMY

#COSINE SIMILARITY
@st.cache
def get_recommendation(title, cosine_sim_mat, df,num_of_rec=10):

    #indices of the course
    course_indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

    #index of the course
    idx = course_indices[title]

    #looking into cosine matrix for that index
    sim_score= list(enumerate(cosine_sim_mat[idx]))
    sim_score = sorted(sim_score, key=lambda x:x[1], reverse=True)
    selected_course_indices=[i[0] for i in sim_score[1:]]
    selected_course_scores = [i[0] for i in sim_score[1:]]
    result_df =df.iloc[selected_course_indices]
    result_df['similarity_score']=selected_course_scores
    final_rec_course= result_df[['Title','similarity_score','Link', 'Stars', 'Rating']]

    return final_rec_course.head(num_of_rec)

# #K MEAREST NEIGHBOR
# @st.cache
# def get_recommendationKNN(title, df,model,num_of_rec=10):
#
#
#
#     model.fit(mainframe)
#     idx=process.extract0ne
#     distances, indices=model.kneighbors(df[idx],n_neighbors=num_of_rec)
#     for i in indices:
#         final=mainframe[['Title', 'Link', 'Stars', 'Rating']]
#     return final







#WEIGHTED AVERAGE
@st.cache
def get_recommendationWA(title, df,num_of_rec=10):

    mainframe =df[['Title', 'Link', 'Stars', 'Rating']]

    v=mainframe['Rating']
    R=mainframe['Stars']
    C=mainframe['Stars'].mean()
    m=mainframe['Rating'].quantile(0.70)
    mainframe['Weighted Average']=(R*v)+(C*m)/(v+m)
    sorted_course=mainframe.sort_values('weighted_average',ascending=False)
    final_rec_course=sorted_course[['Title','weighted_average','Link', 'Stars', 'Rating']]

    return final_rec_course.head(num_of_rec)





@st.cache
def search_term_if_not_found(term,df,num_of_rec=10):
    result_df=df[df['Title'].str.contains(term)]
    rec_course=result_df[['Title','Link', 'Stars', 'Rating']]
    return rec_course.head(num_of_rec)

@st.cache
def search_term_if_not_foundWA(term,df,num_of_rec=10):
    result_df=df[df['Title'].str.contains(term)]
    mainframe=result_df[['Title','Link', 'Stars', 'Rating']]
    v = mainframe['Rating']
    R = mainframe['Stars']
    C = mainframe['Stars'].mean()
    m = mainframe['Rating'].quantile(0.70)
    mainframe['Weighted Average'] = (R * v) + (C * m) / (v + m)
    mainframe=mainframe[['Title','Weighted Average','Link', 'Stars', 'Rating']]
    final_rec_course = mainframe.sort_values('Weighted Average', ascending=False)
    return final_rec_course.head(num_of_rec)


RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #DDDDDD;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">🎓Stars:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑Students:</span>{}</p>
</div>
"""

RESULT_TEMPWA = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #DDDDDD;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Weighted Average::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">🎓Stars:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑Students:</span>{}</p>
</div>
"""
RESULT_TEMP1 = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #DDDDDD;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">🎓Stars:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑Students:</span>{}</p>
</div>
"""





#COURSERA

@st.cache
def get_recommendation_coursera(title, cosine_sim_mat, df,num_of_rec=10):

    #indices of the course
    course_indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

    #index of the course
    idx = course_indices[title]

    #looking into cosine matrix for that index
    sim_score= list(enumerate(cosine_sim_mat[idx]))
    sim_score = sorted(sim_score, key=lambda x:x[1], reverse=True)
    selected_course_indices=[i[0] for i in sim_score[1:]]
    selected_course_scores = [i[0] for i in sim_score[1:]]
    result_df =df.iloc[selected_course_indices]
    result_df['similarity_score']=selected_course_scores
    final_rec_course= result_df[['Title','similarity_score', 'Stars', 'Rating']]
    return final_rec_course.head(num_of_rec)


@st.cache
def search_term_if_not_found_coursera(term,df,num_of_rec=10):
    result_df=df[df['Title'].str.contains(term)]
    rec_course=result_df[['Title', 'Stars', 'Rating']]
    return rec_course.head(num_of_rec)



RESULT_TEMP_coursera1 = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #DDDDDD;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🎓Stars:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑Students:</span>{}</p>
</div>
"""
RESULT_TEMP_Cousera2 = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #DDDDDD;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">🎓Stars:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑Students:</span>{}</p>
</div>
"""



#PROJECTS
@st.cache
def get_recommendation_projects(title, cosine_sim_mat, df,num_of_rec=10):

    #indices of the course
    course_indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

    #index of the course
    idx = course_indices[title]

    #looking into cosine matrix for that index
    sim_score= list(enumerate(cosine_sim_mat[idx]))
    sim_score = sorted(sim_score, key=lambda x:x[1], reverse=True)
    selected_course_indices=[i[0] for i in sim_score[1:]]
    selected_course_scores = [i[0] for i in sim_score[1:]]
    result_df =df.iloc[selected_course_indices]
    result_df['similarity_score']=selected_course_scores
    final_rec_course= result_df[['Title','similarity_score','Link']]

    return final_rec_course.head(num_of_rec)


@st.cache
def search_term_if_not_found_project(term,df,num_of_rec=10):
    result_df=df[df['Title'].str.contains(term)]
    rec_course=result_df[['Title', 'Links']]
    return rec_course.head(num_of_rec)

RESULT_TEMP_project1 = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #DDDDDD;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
</div>
"""

RESULT_TEMP_project2 = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #DDDDDD;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
</div>
"""








#TEST SERIES














#MAIN FUNCTION
def main():
    st.title("RECLEARN: Ai E-Learning App")

    about = ["About the Project", "The Cool Interns", "Temporary"]
    choice = st.sidebar.selectbox("Want to know about us?", about)
    if choice=="About the Project":
        st.subheader("About")
        st.text("Hey There, this is a project made by 4 outstanding interns at IBM \nwhich recommends:""\n"
                "-The Courses best suited for you""\n"
                "-The Projects which'll make you stand out from the crowd""\n"
                "-The test series which help you realise the level you're at""\n")
        st.text(
            "Note: These are all recommendations of the best possible website \nand has been trained on a very small dataset"
            " \nWe'll update the dataset if IBM hires us XD")
    elif choice=="Temporary":
        st.text("Hello, idk why this page is made :/")
    else:
        st.subheader("Contact")
        st.text("We'll attach the official IBM email id's once they hire us\nBut for now, we can only tell our names :p\n"
                "\nSiddhivinayak Dubey\nYash Bharadwaj\nKashish Khurana\nJaswant Singh")
        st.text("Mentored by the very great \nDr. Bhupesh Deewangan(UPES) \nDr. Manish Jain(IBM)")



    menu= ["Courses", "Projects", "Test Series"]
    choice = st.sidebar.selectbox("What do you need us to recommend to?", menu)

    if choice=="Courses":
        st.subheader("Course Recommendation")
        websites = ["Udemy", "Coursera", "Pluralsight", "Geek For Geeks"]
        choice = st.sidebar.selectbox("Select the website you are comfortable with", websites)
        st.text("Type any of your liked courses from udemy and we'll recommend \nthe the best course to you"
                "\nor\n"
                "just type the domain name and we'll try to recommend the mass liked courses")
        search_term = st.text_input("Search")




        #UDEMY
        if choice=="Udemy":
            st.subheader("Udemy Courses")
            algorithm = ["Cosine Similarity","Weighted Average"]
            choice = st.sidebar.selectbox("Optional algo's for nerds",algorithm)

            #COSINE SIMILARITY OUTPUT
            if choice=="Cosine Similarity":
                df = load_data("data/udemy_tech.csv")
                cosine_sim_mat= vectorize_text_to_cosine_mat(df['Title'])
                num_of_rec=st.sidebar.number_input("Number",4,30,7)
                if st.button("Recommend"):
                    if search_term is not None:
                        try:
                            results= get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
                            for row in results.iterrows():
                                rec_title = row[1][0]
                                rec_score = row[1][1]
                                rec_link = row[1][2]
                                rec_star = row[1][3]
                                rec_rating = row[1][4]
                                # st.write("Title",rec_title)
                                stc.html(RESULT_TEMP.format(rec_title, rec_score, rec_link, rec_star, rec_rating),
                                         height=250)
                        except:
                            results = "Hmm seems like you are searching through domains"
                            st.warning(results)
                            st.info("Here's our recommendation for the same :)")

                            result_df= search_term_if_not_found(search_term,df,num_of_rec)
                            # st.dataframe(result_df)
                            for row in result_df.iterrows():
                                rec_title = row[1][0]
                                rec_link = row[1][1]
                                rec_star = row[1][2]
                                rec_rating = row[1][3]
                                # st.write("Title",rec_title)
                                stc.html(RESULT_TEMP1.format(rec_title, rec_link, rec_star, rec_rating),
                                         height=250)
            # #K NEAREST OUTPUT
            # elif choice=="K Nearest":
            #     df = load_data("data/udemy_tech.csv")
            #     num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)
            #     if st.button("Recommend"):
            #         if search_term is not None:
            #             try:
            #                 mainframe = df[['Title', 'Link', 'Stars', 'Rating']]
            #                 students = mainframe.pivot(index='title', columns='Ratings', values='Stars').fillna(0)
            #                 mat_course = csr_matrix(students.values)
            #                 model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
            #                 model_knn.fit(mat_course)
            #                 results = get_recommendationKNN(search_term, df,model_knn, num_of_rec)
            #                 for row in results.iterrows():
            #                     rec_title = row[1][0]
            #                     rec_score = row[1][1]
            #                     rec_link = row[1][2]
            #                     rec_star = row[1][3]
            #                     rec_rating = row[1][4]
            #                     # st.write("Title",rec_title)
            #                     stc.html(RESULT_TEMP.format(rec_title, rec_score, rec_link, rec_star, rec_rating),
            #                              height=250)
            #             except:
            #                 results = "Hmm seems like you are searching through domains"
            #                 st.warning(results)
            #                 st.info("Here's our recommendation for the same :)")
            #
            #                 result_df = search_term_if_not_found(search_term, df, num_of_rec)
            #                 # st.dataframe(result_df)
            #                 for row in result_df.iterrows():
            #                     rec_title = row[1][0]
            #                     rec_link = row[1][1]
            #                     rec_star = row[1][2]
            #                     rec_rating = row[1][3]
            #                     # st.write("Title",rec_title)
            #                     stc.html(RESULT_TEMP1.format(rec_title, rec_link, rec_star, rec_rating),
            #                              height=250)
            #WEIGHTED AVERAGE
            else:
                    df = load_data("data/udemy_tech.csv")
                    num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)
                    if st.button("Recommend"):
                        if search_term is not None:
                            try:
                                results = get_recommendationWA(search_term, df, num_of_rec)
                                for row in results.iterrows():
                                    rec_title = row[1][0]
                                    rec_score = row[1][1]
                                    rec_link = row[1][2]
                                    rec_star = row[1][3]
                                    rec_rating = row[1][4]
                                    # st.write("Title",rec_title)
                                    stc.html(RESULT_TEMP.format(rec_title, rec_score, rec_link, rec_star, rec_rating),
                                             height=250)
                            except:
                                st.info("Here's our recommendation according to the weighted average algorithm :)")

                                result_df = search_term_if_not_foundWA(search_term, df, num_of_rec)
                                # st.dataframe(result_df)
                                for row in result_df.iterrows():
                                    rec_title = row[1][0]
                                    rec_score=row[1][1]
                                    rec_link = row[1][2]
                                    rec_star = row[1][3]
                                    rec_rating = row[1][4]
                                    # st.write("Title",rec_title)
                                    stc.html(RESULT_TEMPWA.format(rec_title,rec_score, rec_link, rec_star, rec_rating),
                                             height=250)


        #COURSERA
        elif choice=="Coursera":
            st.subheader("Coursera Courses")
            df = load_data("data/coursera_data.csv")
            cosine_sim_mat= vectorize_text_to_cosine_mat(df['Title'])
            num_of_rec=st.sidebar.number_input("Number",4,30,7)

            if st.button("Recommend"):

                if search_term is not None:
                    try:
                        results= get_recommendation_coursera(search_term, cosine_sim_mat, df, num_of_rec)
                        for row in results.iterrows():
                            rec_title = row[1][0]
                            rec_score = row[1][1]
                            rec_star = row[1][2]
                            rec_rating = row[1][3]
                            # st.write("Title",rec_title)
                            stc.html(RESULT_TEMP_coursera1.format(rec_title, rec_score, rec_star, rec_rating),
                                     height=250)
                    except:
                        results = "Hmm seems like you are searching through domains"
                        st.warning(results)
                        st.info("Here's our recommendation for the same :)")

                        result_df= search_term_if_not_found_coursera(search_term,df,num_of_rec)
                        # st.dataframe(result_df)
                        for row in result_df.iterrows():
                            rec_title = row[1][0]
                            rec_star = row[1][1]
                            rec_rating = row[1][2]
                            # st.write("Title",rec_title)
                            stc.html(RESULT_TEMP_Cousera2.format(rec_title, rec_star, rec_rating),height=250)


                    #st.write(result)






    #PROJECTS RECOMMENDATIONS
    elif choice=="Projects":
        st.subheader("Project Recommendations")
        websites = ["Geek For Geeks", "CleverProgrammer", "Nevonprojects"]
        choice = st.sidebar.selectbox("Select the website you are comfortable with", websites)
        st.text("Type any of your liked courses from udemy and we'll recommend \nthe the best course to you"
                "\nor\n"
                "just type the domain name and we'll try to recommend the mass liked courses")
        search_term = st.text_input("Search")

        #GEEKFORGEEKS
        if choice=="Geek For Geeks":
            st.subheader("Geek for geeks Projects")
            df = load_data("data/geeksforgeeks.csv")
            cosine_sim_mat = vectorize_text_to_cosine_mat(df['Title'])
            num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)
            if st.button("Recommend"):
                if search_term is not None:
                    try:
                        results = get_recommendation_projects(search_term, cosine_sim_mat, df, num_of_rec)
                        for row in results.iterrows():
                            rec_title = row[1][0]
                            rec_score = row[1][1]
                            rec_link = row[1][2]
                            # st.write("Title",rec_title)
                            stc.html(RESULT_TEMP_project2.format(rec_title, rec_score, rec_link),
                                     height=250)
                    except:
                        results = "Yaay!, you finally decided to level up your game. Here are the best project recommendations for the same"
                        st.warning(results)

                        result_df = search_term_if_not_found_project(search_term, df, num_of_rec)
                        # st.dataframe(result_df)
                        for row in result_df.iterrows():
                            rec_title = row[1][0]
                            rec_link = row[1][1]
                            # st.write("Title",rec_title)
                            stc.html(RESULT_TEMP_project1.format(rec_title, rec_link,),
                                     height=150)
        #CLEVERPROGRAMMER
        else:
            st.subheader("Clever Programmer Courses")
            df = load_data("data/thecleverprogrammer.csv")
            cosine_sim_mat = vectorize_text_to_cosine_mat(df['Title'])
            num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)
            if st.button("Recommend"):
                if search_term is not None:
                    try:
                        results = get_recommendation_projects(search_term, cosine_sim_mat, df, num_of_rec)
                        for row in results.iterrows():
                            rec_title = row[1][0]
                            rec_score = row[1][1]
                            rec_link = row[1][2]
                            rec_star = row[1][3]
                            rec_rating = row[1][4]
                            # st.write("Title",rec_title)
                            stc.html(RESULT_TEMP_project2.format(rec_title, rec_score, rec_link, rec_star, rec_rating),
                                     height=250)
                    except:
                        results = "Yaay!, you finally decided to level up your game. Here are the best project recommendations for the same"
                        st.warning(results)

                        result_df = search_term_if_not_found_project(search_term, df, num_of_rec)
                        # st.dataframe(result_df)
                        for row in result_df.iterrows():
                            rec_title = row[1][0]
                            rec_link = row[1][1]
                            # st.write("Title",rec_title)
                            stc.html(RESULT_TEMP_project1.format(rec_title, rec_link,),
                                     height=150)
                # NEVON PROJECTS
                elif choice == "Nevonprojects":
                    st.subheader("Nevon Projects")
                    df = load_data("data/nevonprojects.csv")
                    cosine_sim_mat = vectorize_text_to_cosine_mat(df['Title'])
                    num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)
                    if st.button("Recommend"):
                        if search_term is not None:
                            try:
                                results = get_recommendation_projects(search_term, cosine_sim_mat, df, num_of_rec)
                                for row in results.iterrows():
                                    rec_title = row[1][0]
                                    rec_score = row[1][1]
                                    rec_link = row[1][2]
                                    rec_star = row[1][3]
                                    rec_rating = row[1][4]
                                    # st.write("Title",rec_title)
                                    stc.html(RESULT_TEMP_project2.format(rec_title, rec_score, rec_link, rec_star,
                                                                         rec_rating),
                                             height=250)
                            except:
                                results = "Yaay!, you finally decided to level up your game. Here are the best project recommendations for the same"
                                st.warning(results)


                                result_df = search_term_if_not_found_project(search_term, df, num_of_rec)
                                # st.dataframe(result_df)
                                for row in result_df.iterrows():
                                    rec_title = row[1][0]
                                    rec_link = row[1][1]
                                    # st.write("Title",rec_title)
                                    stc.html(RESULT_TEMP_project1.format(rec_title, rec_link, ),
                                             height=150)


    # TEST RECOMMENDATIONS
    else:
        st.subheader("Test Recommendations")
        websites = ["Sanfoundry", "Courseya"]
        choice = st.sidebar.selectbox("Select the website you are comfortable with", websites)
        st.text("Type the domain of tests you want check on")
        search_term = st.text_input("Search")

        #SANFOUNDRY
        if choice=="Sanfoundry":
            st.subheader("Tests on Sanfoundry")
            df = load_data("data/sanfoundry.csv")
            cosine_sim_mat = vectorize_text_to_cosine_mat(df['Title'])
            num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)
            if st.button("Recommend"):
                if search_term is not None:
                    try:
                        results = get_recommendation_projects(search_term, cosine_sim_mat, df, num_of_rec)
                        for row in results.iterrows():
                            rec_title = row[1][0]
                            rec_score = row[1][1]
                            rec_link = row[1][2]
                            rec_star = row[1][3]
                            rec_rating = row[1][4]
                            # st.write("Title",rec_title)
                            stc.html(RESULT_TEMP_project2.format(rec_title, rec_score, rec_link, rec_star,
                                                                 rec_rating),
                                     height=250)
                    except:
                        results = "Okay, we know you've worked hard for it.. Let's get through this!"
                        st.warning(results)
                        # st.info("Here's our recommendation for the same :)")

                        result_df = search_term_if_not_found_project(search_term, df, num_of_rec)
                        # st.dataframe(result_df)
                        for row in result_df.iterrows():
                            rec_title = row[1][0]
                            rec_link = row[1][1]
                            # st.write("Title",rec_title)
                            stc.html(RESULT_TEMP_project1.format(rec_title, rec_link, ),
                                     height=150)

        #Courseya
        else:
            st.subheader("Courseya Tests")
            df = load_data("data/courseya.csv")
            cosine_sim_mat = vectorize_text_to_cosine_mat(df['Title'])
            num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)
            if st.button("Recommend"):
                if search_term is not None:
                    try:
                        results = get_recommendation_projects(search_term, cosine_sim_mat, df, num_of_rec)
                        for row in results.iterrows():
                            rec_title = row[1][0]
                            rec_score = row[1][1]
                            rec_link = row[1][2]
                            rec_star = row[1][3]
                            rec_rating = row[1][4]
                            # st.write("Title",rec_title)
                            stc.html(RESULT_TEMP_project2.format(rec_title, rec_score, rec_link, rec_star,
                                                                 rec_rating),
                                     height=250)
                    except:
                        results = "Okay, we know you've worked hard for it.. Let's get through this!"
                        st.warning(results)
                        # st.info("Here's our recommendation for the same :)")

                        result_df = search_term_if_not_found_project(search_term, df, num_of_rec)
                        # st.dataframe(result_df)
                        for row in result_df.iterrows():
                            rec_title = row[1][0]
                            rec_link = row[1][1]
                            # st.write("Title",rec_title)
                            stc.html(RESULT_TEMP_project1.format(rec_title, rec_link, ),
                                     height=150)









if __name__ == '__main__':
    main()
