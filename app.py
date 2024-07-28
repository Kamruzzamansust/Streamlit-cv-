import streamlit as st
from  streamlit_option_menu import  option_menu 
from streamlit_lottie import st_lottie
import json
from function.fun import project_overview,markdown_writting
st.set_page_config(layout="wide")
# Set the title of the web app

import streamlit as st
import pandas as pd

# Define the data
skills_data = [
    ["Programming Language", "Python, R"],
    ["Spreadsheet", "Excel, Google Sheet"],
    ["OS", "Windows, Linux"],
    ["Database", "SQL (PostgreSQL)"],
    ["Data Analysis and Visualization", "Pandas, Numpy, Seaborn, Matplotlib, dplyr, Tidyverse, ggplot2"],
    ["Design and front-end Development", "plotly-dash, R-Shiny, Streamlit"],
    ["Machine Learning Framework", "Scikit-Learn, Tensorflow, Keras"],
    ["API Development", "FastAPI"],
    ["Containerization", "Docker"],
    ["Big Data Technology", "Pyspark, Dask"],
    ["BI Tool", "Power BI"],
    ["Time Series", "Facebook Prophet"],
    ["NLP Related Technology", "Hugging Face, Langchain, Fine-Tuning techniques, Transfer Learning"],
    ["Model Experiment,Tracking, Packaging", "MLflow"],
    ["Version Control", "Git, Github"]
]

# Convert to DataFrame
df = pd.DataFrame(skills_data, columns=["Field", "Tools"])

# Display the table in Streamlit


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")


lottie_coder = load_lottiefile("Animation - 1721204539631.json")
lottie_contact = load_lottiefile("contact.json")

lotte_skill = load_lottiefile("skill.json")

st.title("Welcome To My Portfolio !!")
st.write("----")

# Sidebar with picture
st.sidebar.image(r"arif2.png",width = 4, use_column_width=True)

st.markdown("""
<style>
.sidebar .sidebar-content {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h2 style='text-align: center;'>Md Kamruzzaman</h2>", unsafe_allow_html=True)

with st.sidebar.container(border=True):
    selected = option_menu(
        menu_title = None,
        options =['About',"Career Interest",'Skills','Projects','Contact','Get in Touch !!'],
        icons=['file-person-fill',"check2-circle",'hammer', 'code-slash','chat-left-text-fill','magnet-fill']
    )

st.markdown("""
    <script>
    function scrollToSection(sectionId) {
        document.getElementById(sectionId).scrollIntoView({ behavior: 'smooth' });
    }
    </script>
    """, unsafe_allow_html=True)

if selected == "About":

    with st.container(border = True):
        col1,col2 = st.columns(2)
        with col1:
            st.write("##")
            st.write()
            st.write(":wave::wave::wave::wave::wave::wave:")
            st.write("""
    Hi! I'm **Md Kamruzzaman**, an Aspiring data scientist.
    Welcome to my portfolio website.
    Here you can find information about my work, projects, and skills.
    """)
        with open("Md.Kamruzzaman_Resume.pdf", "rb") as file:
            btn = st.download_button(
                label="Download Resume",
                data=file,
                file_name="Md.Kamruzzaman_Resume.pdf",
                mime="application/pdf"
            )

        with col2:
            st_lottie(lottie_coder, height=300,width = 500, key="coder")

    st.write("----")

    with st.container(border = True):
      
            st.subheader("""
                         
          Education 
                         
             - Bsc 
                         
                 - Statistics
                         
                 - Shahlal University of Science and Technology 
                                            
                 - CGPA : 3.16
                         
             - Msc
                         
                 - Applied Statistics and Data Science 
                         
                 - Jahangirnagar Univesrsity
                         
                 - CGPA: 3.77
                         
                                          
                    """)
          


elif selected == "Career Interest":
    with st.container(border = True):
        st.markdown("""
## Career Interest :bow_and_arrow:

With a strong educational background that includes a **BSc in Statistics** and an **MSc in Applied Statistics and Data Science**, I am passionate about leveraging my skills and knowledge in **business, AI, data analysis, and data visualization**. My career goal is to apply advanced statistical techniques and data science methodologies to drive business insights, enhance decision-making processes, and contribute to innovative solutions in the field of artificial intelligence.

I am particularly interested in roles that involve:

- Data-driven strategy development
- Predictive analytics
- Creation of compelling visualizations to communicate complex data insights effectively
""")

elif selected == "Skills":
      st.subheader(" Technical Skills")
      with st.container(border = True):
        
        st_lottie(lotte_skill,height=300,width = 900, key="skills")
        st.write("----")
        st.write("")
    #st.write("")
      st.table(df)



elif selected == "Projects":
    st.title("Projects Overview")
    with st.container(border=True):
        selected2 = option_menu(
                menu_title = None,
                options =['SQL','Power Bi','ML',"DASH",'R-Shiny',"Tensorflow","NLP","Time Series"],
                orientation= 'horizontal'
        )
    
    if selected2 == "Power Bi":
            markdown_writting("**:blue-background[Click to See My Overall  Power BI journey]**","""
### My Power BI Journey

I divided my learning into three phases:

#### 1. Data Modeling
- Learned about dimensions and fact tables
- Explored star, snowflake, and galaxy schemas
- Understood role-playing dimensions and the importance of a date table
- Learned about cardinality

#### 2. ETL through Power Query
- Loaded data from various sources like Excel, CSV, and databases
- Performed data cleaning using built-in Power Query functions
- Explored duplicates and references in Power Query
- Focused on using built-in functionality without M script

#### 3. Data Analysis with DAX
- Learned about DAX functions, including table and relation functions
- Explored calculated columns and measures
- Understood how filters propagate
- Focused on the CALCULATE function and context modifiers
- Learned about various time intelligence functions




""")
            st.write("**<<Power BI Projects>>**")
            with st.container(border = True):
              
             
              with st.container(border = True):
                col1, col2 = st.columns(2)
                with col1:
                    
                    st.write("1. Adventure Works Sales Analysis")
                    #st.write("----")
                    st.video("Power Bi project 2.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/AW-SALES_POWERBI)
                            
                                """)
                    
                with col2:
                    
                    st.markdown("""
        ##### Project Description

        I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

        - **Dynamic Parameters:** Show top customers by sales.
        - **Advanced DAX Functions:** Enhance insights with complex measures.
        - **Context Transition:** Use `CALCULATE()` for precise analysis.

        This project showcases my skills in data visualization and strategic insights.
        """)
                    
              st.write("----")
              with st.container(border = True):
                col1, col2 = st.columns(2)
                with col1:
                    
                    st.write("2. Hospital-Patient Data Analysis ")
                    #st.write("----")
                    st.video(r"healthpowerbi.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/healthcare-data-analysis-Power-bi)
                            
                                """)
                    
                with col2:
                    
                    st.markdown("""
        ##### Project Description

        I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

        - **Dynamic Parameters:** Show top customers by sales.
        - **Advanced DAX Functions:** Enhance insights with complex measures.
        - **Context Transition:** Use `CALCULATE()` for precise analysis.

        This project showcases my skills in data visualization and strategic insights.
        """)
                    
              st.write("----")
              with st.container(border = True):
                col1, col2 = st.columns(2)
                with col1:
                    
                    st.write("2. Transportation Data Analysis ")
                    #st.write("----")
                    st.video(r"Transportation.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/healthcare-data-analysis-Power-bi)
                            
                                """)
                    
                with col2:
                    
                    st.markdown("""
        ##### Project Description

        I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

        - **Dynamic Parameters:** Show top customers by sales.
        - **Advanced DAX Functions:** Enhance insights with complex measures.
        - **Context Transition:** Use `CALCULATE()` for precise analysis.

        This project showcases my skills in data visualization and strategic insights.
        """)
                    
              st.write("----")
              with st.container(border = True):
                col1, col2 = st.columns(2)
                with col1:
                    
                    st.write('4. "DV RENTAL" Data Base Analysis Using DAX ')
                    #st.write("----")
                    st.video(r"DVDrental.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/healthcare-data-analysis-Power-bi)
                            
                                """)
                    
                with col2:
                    
                    st.markdown("""
        ##### Project Description

        I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

        - **Dynamic Parameters:** Show top customers by sales.
        - **Advanced DAX Functions:** Enhance insights with complex measures.
        - **Context Transition:** Use `CALCULATE()` for precise analysis.

        This project showcases my skills in data visualization and strategic insights.
        """)
    
    elif selected2 == "SQL":
        
        markdown_writting("**:blue-background[Click to See My Overall SQL journey]**","""
                            


### My SQL Journey

When I first learned SQL, I discovered various SQL commands like:

- **DDL**: Data Definition Language
- **DQL**: Data Query Language
- **DML**: Data Manipulation Language
- **DCL**: Data Control Language
- **TCL**: Transaction Control Language

Then, I explored database keys, including primary keys, foreign keys, composite keys, surrogate keys, and candidate keys. I also tried to understand schema design and the concepts of data warehouses, data marts, and data lakes.

After grasping the theoretical concepts, I began learning about SQL queries. I focused on:

- Basic query execution order and clauses
- Various constraints used in SQL
- Joins, such as:
  - Inner join
  - Left and right outer join
  - Full outer join
  - Natural join
  - Self join

I then learned about subqueries, including:

- Subqueries with FROM
- Subqueries with SELECT
- Subqueries with WHERE and IN

Next, I combined these with joins and explored CTEs (Common Table Expressions). I also learned about various window functions, transactions, and indexing methods like B-tree, hash, BRIN, and partial indexing to speed up queries.

This is my overall SQL journey.


                              """)
        st.write("**<<SQL Projects>>**")
        
        # Main container
        with st.container():
            # Inner container
            with st.container():
                # Single column
                col1 = st.columns(1)[0]
                with col1:
                    st.write("1. DVD RENTAL DATABASE ANALYSIS in POSTGRESQL")
                    st.video(r"dvrentaldatabsesql.mp4")
                    st.write("---")
                    st.markdown("[GitHub link](https://github.com/Kamruzzamansust/DVD-rental-Sql-Data-analysis-Project/blob/main/dvdrental_project.sql)")
                    st.write('---')
            with st.container(border = True):
                st.markdown("""
            ##### Project Description

            I used :orange-background[DVD rental database] Using Postgresql to create a SQL script to analyze the data. Key features:

            - Apply Basic Query
            - Uses Of  Join and Subquery 
            - Some advance Techniques (Window Function , CTE)

           
            """)
        
            

















    elif selected2 == "ML":
        markdown_writting("**:blue-background[Click to See  My Overall  Machine Learning journey]**","""
        ### My Machine Learning Journey

#### Data Cleaning and Visualization
- Learned data cleaning techniques with Pandas and NumPy
- Used Matplotlib and Seaborn for data visualization
  - **Univariate Analysis**: Count plot, pie plot, histogram, distplot, box plot
  - **Bivariate Analysis**: Scatter plot, bar plot, box plot, heatmap, pairplot, line plot
  - **Multivariate Analysis**: Explored complex relationships between features

#### Feature Engineering
- **Feature Transformation**:
  - Missing value handling: Mean, median, mode imputation, KNN imputation, regression imputation
  - Handling categorical values and outlier detection
  - Feature scaling: Min-max scaling, Z-normalization
- **Feature Construction and Selection**:
  - Techniques: Correlation coefficient, chi-square test, information gain
- **Feature Extraction**:
  - Methods: PCA, LDA, t-SNE

#### Understanding ML Algorithms (Supervise Machine Laearning )
- **Regression**:
  - Simple Linear, Multiple Linear, and polynomial regression
  - Regularization techniques: Lasso, ridge regression
  - Gradient descent for finding global minima
- **Classification**:
  - Algorithms: SVM with kernel tricks, KNN, Naive Bayes
  - Decision Trees: Information gain and entropy
  - Importance of decision trees in ensemble methods: Random forest, bagging, stacking, boosting
  - Weak learners in boosting algorithms

#### Hyperparameter Tuning and Validation
- Explored hyperparameters associated with various algorithms
- Validation methods:
  - Cross-validation and K-fold techniques
  - Validation metrics: RMSE, MSE for regression, confusion matrix for classification

####Clustering (Unsuporvised Machine Learning)
- K Means , Kmeans ++
- Hierarchical clustering
- DBSCAN clustering - helpful for non linear clustering
                           
**Performance matrics for Clustering**
   - silhoutte Score                                                                           

#### Additional Concepts

- How distrbution changes over Time which is known as concept drift 
                          
                          


                          """)
        #st.write("**<<Machhine Learning Projects>>**")
        st.write("**<<Machine Learning Projects>>**")
        with st.container(border = True):
              
             
              with st.container(border = True):
                col1, col2 = st.columns(2)
                with col1:
                    
                    st.write("**1. Food Delivery Status Classification with Xg BOOST and Streamlit** ")
                    #st.write("----")
                    st.video(r"Food.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/AW-SALES_POWERBI)
                            
                                """)
                    
                with col2:
                    
                    st.markdown("""
        ##### Project Description

        I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

        - **Dynamic Parameters:** Show top customers by sales.
        - **Advanced DAX Functions:** Enhance insights with complex measures.
        - **Context Transition:** Use `CALCULATE()` for precise analysis.

        This project showcases my skills in data visualization and strategic insights.
        """)
                    
              st.write("----")
              with st.container(border = True):
                 col1, col2 = st.columns(2)
                 with col1:
                    
                    st.write("**2. Customer Segmentation Ml Classification Problem** ")
                    #st.write("----")
                    st.video(r"customer segmentation.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/AW-SALES_POWERBI)
                            
                                """)
                    
                 with col2:
                    
                    st.markdown("""
        ##### Project Description

        I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

        - **Dynamic Parameters:** Show top customers by sales.
        - **Advanced DAX Functions:** Enhance insights with complex measures.
        - **Context Transition:** Use `CALCULATE()` for precise analysis.

        This project showcases my skills in data visualization and strategic insights.
        """)
              st.write("----")
              with st.container(border = True):
                 col1, col2 = st.columns(2)
                 with col1:
                    
                    st.write("**3. Customer Churn Analysis with Logistic Regression** ")
                    #st.write("----")
                    st.video(r"customer_churn_analysis.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/AW-SALES_POWERBI)
                            
                                """)
                    
                 with col2:
                    
                    st.markdown("""
        ##### Project Description

        I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

        - **Dynamic Parameters:** Show top customers by sales.
        - **Advanced DAX Functions:** Enhance insights with complex measures.
        - **Context Transition:** Use `CALCULATE()` for precise analysis.

        This project showcases my skills in data visualization and strategic insights.
        """)
              st.write("----")
              with st.container(border = True):
                        col1, col2 = st.columns(2)
                        with col1:
                            
                            st.write("**4. Customer Life Time Value Analysis With RFM** ")
                            #st.write("----")
                            st.video(r"CLTV.mp4")
                            st.write("---")
                            st.markdown("""
                                    [GitHub link](https://github.com/Kamruzzamansust/AW-SALES_POWERBI)
                                    
                                        """)
                            
                        with col2:
                            
                            st.markdown("""
                ##### Project Description

                I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

                - **Dynamic Parameters:** Show top customers by sales.
                - **Advanced DAX Functions:** Enhance insights with complex measures.
                - **Context Transition:** Use `CALCULATE()` for precise analysis.

                This project showcases my skills in data visualization and strategic insights.
                """)
                     
              st.write("----")
              with st.container(border = True):
                            col1, col2 = st.columns(2)
                            with col1:
                                        
                                        st.write("**5. Regression and Clustering problem with pyspark** ")
                                        #st.write("----")
                                        st.video(r"ml_pyspark.mp4")
                                        st.write("---")
                                        st.markdown("""
                                                [GitHub link](https://github.com/Kamruzzamansust/AW-SALES_POWERBI)
                                                
                                                    """)
                                        
                            with col2:
                                        
                                        st.markdown("""
                            ##### Project Description

                            I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

                            - **Dynamic Parameters:** Show top customers by sales.
                            - **Advanced DAX Functions:** Enhance insights with complex measures.
                            - **Context Transition:** Use `CALCULATE()` for precise analysis.

                            This project showcases my skills in data visualization and strategic insights.
                            """)
              st.write("---")
              with st.container(border = True):
                            col1, col2 = st.columns(2)
                            with col1:
                                        
                                        st.write("**6. Market Busket Analysis with Python Aripri Algorithm** ")
                                        #st.write("----")
                                        st.video(r"mba.mp4")
                                        st.write("---")
                                        st.markdown("""
                                                [GitHub link](https://github.com/Kamruzzamansust/AW-SALES_POWERBI)
                                                
                                                    """)
                                        
                            with col2:
                                        
                                        st.markdown("""
                            ##### Project Description

                            I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

                            - **Dynamic Parameters:** Show top customers by sales.
                            - **Advanced DAX Functions:** Enhance insights with complex measures.
                            - **Context Transition:** Use `CALCULATE()` for precise analysis.

                            This project showcases my skills in data visualization and strategic insights.
                            """)
              st.write("----")                            
              with st.container(border = True):
                            col1, col2 = st.columns(2)
                            with col1:
                                        
                                        st.write("**7. Using Apriori Algorithm For Coffe Recommendation** ")
                                        #st.write("----")
                                        st.video(r"apriori.mp4")
                                        st.write("---")
                                        st.markdown("""
                                                [GitHub link](https://github.com/Kamruzzamansust/AW-SALES_POWERBI)
                                                
                                                    """)
                                        
                            with col2:
                                        
                                        st.markdown("""
                            ##### Project Description

                            I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

                            - **Dynamic Parameters:** Show top customers by sales.
                            - **Advanced DAX Functions:** Enhance insights with complex measures.
                            - **Context Transition:** Use `CALCULATE()` for precise analysis.

                            This project showcases my skills in data visualization and strategic insights.
                            """)
                                        

                        
                   
                    
                    
                                
 

                        


    elif selected2 == "Tensorflow":
        st.write("**<<Tensorflow Learning>**")
        
        # Main container
        with st.container():
            # Inner container
            with st.container():
                # Single column
                col1 = st.columns(1)[0]
                with col1:
                    st.write("1.Hands On Tensorflow and Keras")
                    st.video(r"Tensorflow.mp4")
                    st.write("---")
                    st.markdown("[GitHub link](https://github.com/Kamruzzamansust/AW-SALES_POWERBI)")
                    st.write('---')
            with st.container(border = True):
                st.markdown("""
            ##### Project Description

            I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

            - **Dynamic Parameters:** Show top customers by sales.
            - **Advanced DAX Functions:** Enhance insights with complex measures.
            - **Context Transition:** Use `CALCULATE()` for precise analysis.

            This project showcases my skills in data visualization and strategic insights.
            """)
                    
              #st.write("----")
             

                        
    elif selected2 == "R-Shiny":
        st.write("**<<R-Shiny Projects>>**")
        with st.container(border = True):
              
             
              with st.container(border = True):
                col1, col2 = st.columns(2)
                with col1:
                    
                    st.write("**1. House For Rent R-Shiny Web App** ")
                    #st.write("----")
                    st.video(r"mapping_record.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/AW-SALES_POWERBI)
                            
                                """)
                    
                with col2:
                    
                    st.markdown("""
        ##### Project Description

        I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

        - **Dynamic Parameters:** Show top customers by sales.
        - **Advanced DAX Functions:** Enhance insights with complex measures.
        - **Context Transition:** Use `CALCULATE()` for precise analysis.

        This project showcases my skills in data visualization and strategic insights.
        """)
                    
              st.write("----")
              with st.container(border = True):
                 col1, col2 = st.columns(2)
                 with col1:
                    
                    st.write("**2. Netflix Data Visualisation R-Shiny** ")
                    #st.write("----")
                    st.video(r"R-shiny-netflix.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/AW-SALES_POWERBI)
                            
                                """)
                    
                 with col2:
                    
                    st.markdown("""
        ##### Project Description

        I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

        - **Dynamic Parameters:** Show top customers by sales.
        - **Advanced DAX Functions:** Enhance insights with complex measures.
        - **Context Transition:** Use `CALCULATE()` for precise analysis.

        This project showcases my skills in data visualization and strategic insights.
        """)
                    
  
             




    elif selected2 == "DASH": 
        #st.write("**<<Machhine Learning Projects>>**")
        st.write("**<<Plotly-Dash Projects>>**")
        with st.container(border = True):
              
             
              with st.container(border = True):
                col1, col2 = st.columns(2)
                with col1:
                    
                    st.write("1. IPL Data Analysis with Python Plotly Dash  ")
                    #st.write("----")
                    st.video(r"ipl.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/AW-SALES_POWERBI)
                            
                                """)
                    
                with col2:
                    
                    st.markdown("""
        ##### Project Description

        I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

        - **Dynamic Parameters:** Show top customers by sales.
        - **Advanced DAX Functions:** Enhance insights with complex measures.
        - **Context Transition:** Use `CALCULATE()` for precise analysis.

        This project showcases my skills in data visualization and strategic insights.
        """)
                    
              st.write("----")
              with st.container(border = True):
                col1, col2 = st.columns(2)
                with col1:
                    
                    st.write("2. Adventure works Sales Data Visualization with Python Plotly Dash ")
                    #st.write("----")
                    st.video(r"Dash.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/healthcare-data-analysis-Power-bi)
                            
                                """)
                    
                with col2:
                    
                    st.markdown("""
        ##### Project Description

        I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

        - **Dynamic Parameters:** Show top customers by sales.
        - **Advanced DAX Functions:** Enhance insights with complex measures.
        - **Context Transition:** Use `CALCULATE()` for precise analysis.

        This project showcases my skills in data visualization and strategic insights.
        """)
              st.write("----")
                    
              with st.container(border = True):
                 col1, col2 = st.columns(2)
                 with col1:
                    
                    st.write("2. Youtube  Data Visualization with Python Plotly Dash ")
                    #st.write("----")
                    st.video(r"Youtube.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/healthcare-data-analysis-Power-bi)
                            
                                """)
                    
                 with col2:
                    
                    st.markdown("""
                                
        #### Project Description

        I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

        - **Dynamic Parameters:** Show top customers by sales.
        - **Advanced DAX Functions:** Enhance insights with complex measures.
        - **Context Transition:** Use `CALCULATE()` for precise analysis.

        This project showcases my skills in data visualization and strategic insights.
        """)
                    

    elif selected2 == "NLP": 
        #st.write("**<<Machhine Learning Projects>>**")
        st.write("**<<NLP Projects>>**")
        with st.expander("**NLP Basics**"):
             
        #st.write("**<<NLP Projects>>**")
            #with st.container(border = True):
                project_overview("1. Basic NLP ",
                                r"NLP1.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/1CQH2w15evItQVXFQqI5NpvjGYiVboAhf#scrollTo=P6IronP9qrg7)",
                                """
                                    
           
            """

                                )
                st.write('---')
                project_overview("2. Understanding Transformers ",
                                r"NLP2.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/1DO1H39HiFiFmU1Z-uJaYwlENEpxqoAav#scrollTo=8t1w7p8cu3Al)",
                                """
                                    
           
            """)
                
                st.write('---')
                project_overview("3. Architecture Development Of Transformers",
                                 r"NLP3.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/1dLbRS5BPRyswrhkv0wv7WcvFgcPRC3hP#scrollTo=fnnfAaKCIcDt)",
                                """
                                    
           
            """)
        with st.expander("**NLP INtermediate**"):
              st.write('---')
              project_overview("1. **Trying To Know about BERT**",
                                 r"BERT1.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/1edixmSMEUSPj6eO5V1_xLExkYwgeqQIr#scrollTo=qQDChvozM1QJ)",
                                """
                                    
           
            """)
              st.write("---")
              project_overview("2. **Masked Language modeling and Next Sentence Prediction With BERT** ",
                                 r"BERTMASKED.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/18lG5UvQ8N1djCeHDwXV9d6zEUhnYgoyQ#scrollTo=qdHfE77rCBLs)",
                                """
                                    
           
            """)
              st.write("---")
              project_overview("3. **BERT Fine Tuning ( Without Pipeline )** ",
                                 r"BERT2.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/1rM0prvLLajG81bZew8bhykjTtiQ2pqEU#scrollTo=iravzBml_Dvu)",
                                """
                                    
           
            """)
              st.write("---")
              project_overview("4. **Fine Tuning DistilBERT With Custom Datest** ",
                                 r"BERT3.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/1qvcFj94l4-IrQ17X2tkoMwdE4RV8e96l#scrollTo=3uJ90BUf9gU1)",
                                """
                                    
           
            """)
              st.write("---")
              project_overview("5. **Create Api With Fast Api for Streamlit and httr for R shiny** ",
                                 r"finetune.mp4",
                                "[LINK NEED]",
                                """
                                    
           
            """)
              st.write("---")
              project_overview("6. **Trying To understand QA system** ",
                                 r"QA.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/1LrWV8ugDtLqozeNmjnVWnjGgraAG3eq0#scrollTo=EhCpzkFcb0mW)",
                                """
                                    
           
            """)
              st.write("---")
              project_overview("7. Text Summarization Transfer Learning With GPT-2, google-pegasus,T5",
                                 r"Text_Summarization_simple.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/1WNG2ZVvSBpuK3ZBEFCm8b1SV1Y1os1NY#scrollTo=LFEy_b0FU4U4)",
                                """
                                    
           
            """)
              st.write("---")
              project_overview("8. **Question Answering With BERT** ",
                                 r"Bert_qa_model.mp4",
                                "As File Size size Large can't push to Github",
                                """
                                    
           
            """)
              st.write("---")
        with st.expander("**NLP Advance**"):
             st.write('---')
             project_overview("1. **BD Constitutional Chatbot** ",
                                 r"con_rag.mp4",
                                 
                                "[Github Link](https://github.com/Kamruzzamansust/bd-con-rag)",
                                """
                                    
           
            """)
             
             
           
             
            
              
      













if selected == "Contact":
    
    with st.container(border = True):
        st.write(":house: House no -11 , Road No -7 , Mirpur, Dhaka,Bangladesh")
        st.write(":phone: XXXXXXXXXXX")
        st.write(":globe_with_meridians: linked In Profile: https://www.linkedin.com/in/md-kamruzzaman-57a60925a/ ")


if selected == "Get in Touch !!":
    #st.title("Get in Touch !!")
    #st.write("#")
    container_style = """
    <style>
    .custom-container {
        border: 2px solid #4CAF50;
        padding: 20px;
        margin-bottom: 20px;
    }
    </style>
    """

    contact_form = """
    <div class="custom-container">
        <form action="https://formsubmit.co/kamruzzaman.sust15@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here" required></textarea>
            <button type="submit">Send</button>
        </form>
    </div>
    """

    st.markdown(container_style, unsafe_allow_html=True)
    left_col, right_col = st.columns((2, 1))
    with left_col:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_col:
        st_lottie(lottie_contact)

  
