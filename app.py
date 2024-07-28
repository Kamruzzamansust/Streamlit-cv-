import streamlit as st
from  streamlit_option_menu import  option_menu 
from streamlit_lottie import st_lottie
import json
from function.fun import project_overview ,markdown_writting
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
  - SV Regression , RF , BAGGING , BOOSTing Algorithms
  - Regularization techniques: Lasso, ridge regression
  - Gradient descent for finding global minima
- **Classification**:
  - Algorithms: Logistic Regression ,SVM classifier with kernel tricks, KNN, Naive Bayes
  - Decision Trees: Information gain and entropy
  - Importance of decision trees in ensemble methods: Random forest, bagging, stacking, boosting
  - Weak learners in boosting algorithms

#### Hyperparameter Tuning and Validation
- Explored hyperparameters associated with various algorithms
- Validation methods:
  - Cross-validation and K-fold techniques
  - Validation metrics: RMSE, MSE for regression, confusion matrix ,AUC for classification

#### Clustering (Unsuporvised Machine Learning)
- K Means , Kmeans ++
- Hierarchical clustering
- DBSCAN and HDBSCAN clustering - helpful for non linear clustering
                           
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
        markdown_writting("**:blue-background[Click to See  My Overall  Deep Learning journey With Tensorflow]**",'''
        ### My Deep Learning Journey with TensorFlow

#### Learning Deep Learning Fundamentals
- **Types of Neural Networks**:
  - Explored various architectures and their applications
- **Perceptron and MLP**:
  - Studied the basics of perceptrons and multilayer perceptrons (MLP)
- **Loss Functions**:
  - Understood how loss functions guide model training
- **Propagation Techniques**:
  - **Forward Propagation**: How inputs are transformed through the network
  - **Backpropagation**: Calculating gradients to update weights
- **Gradient Descent**:
  - Learned how it optimizes the model
  - **Vanishing Gradient Problem**: Identified issues with deep networks and explored solutions

#### Improving Model Performance
- **Techniques**:
  - Early stopping, data scaling, dropout layers, and regularization
- **Activation Functions**:
  - Explored various types, including how ReLU addresses vanishing gradients
- **Weight Initialization**:
  - Techniques to start training effectively
- **Batch Normalization**:
  - Importance in stabilizing and speeding up training
- **Optimizers**:
  - Studied optimizers and their relation to gradient descent

#### Applying with TensorFlow
- Began implementing these concepts using TensorFlow for practical applications

This journey has deepened my understanding of deep learning and equipped me with the skills to use TensorFlow effectively.
                          ''')
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
        markdown_writting("**:blue-background[Click to See  My Overall  R shiny Journey]**","""
        ### My R and R Shiny Journey

#### Learning R
- **Data Manipulation**: 
  - Used `dplyr` and `tidyverse` for efficient data manipulation
- **Handling Variables**:
  - `forcats` for categorical variables
  - `stringr` for string operations
  - `lubridate` for managing date variables
- **Data Visualization**:
  - Created plots using `ggplot2` for basic visualizations

#### Exploring R Shiny
- **UI Components**:
  - Learned about various UI inputs and outputs
- **Reactivity**:
  - Explored how reactivity works in Shiny applications
  - Created reactive expressions and understood how they change based on user input
- **Building Interactive Apps**:
  - Started building simple interactive applications
  - Experimented with layout and design for better user experience

#### Enhancing Shiny Apps
- **Shiny Dashboard**:
  - Learning to use the `shinydashboard` package for structured dashboards
- **Styling**:
  - Exploring `bslib` for customizable themes
  - Using the `fresh` library for advanced styling options

This journey has equipped me with foundational skills in R and R Shiny, setting the stage for creating dynamic and interactive applications.





         """)

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
              st.write("-----")
              project_overview("Simple Sales Dashboard with R shiny",
                               r"Sales_dashbord_R shiny.mp4",
                               "[GitHub link](https://github.com/Kamruzzamansust/AW-SALES_POWERBI)",
                               """
                               ##### Project Description

        I used Adventure Works sales data to create an interactive Power BI dashboard. Key features:

        - **Dynamic Parameters:** Show top customers by sales.
        - **Advanced DAX Functions:** Enhance insights with complex measures.
        - **Context Transition:** Use `CALCULATE()` for precise analysis.

        This project showcases my skills in data visualization and strategic insights.
        """



                               )

            
  
             




    elif selected2 == "DASH": 
        markdown_writting("**:blue-background[Click to See  My Overall  Plotly dash Journey]**","""
        ### My Plotly Dash Journey

#### Dashboard Layout
- Learned how to create and organize dashboards using Dash

#### Plotly Graphs
- Explored various types of plots using Plotly:
  - Line charts
  - Bar charts
  - Scatter plots
  - Pie charts
  - Heatmaps
  - More advanced visualizations

#### Callbacks
- Implemented basic callbacks to make dashboards interactive
- Used callbacks to update graphs dynamically based on user input

This journey helped me create interactive and visually appealing dashboards with Python and Plotly Dash.

         """)




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
        markdown_writting("**:blue-background[Click to See  My Overall NLP journey]**",'''
        ### My NLP Journey: --------->>> stage 1 
#### Text Preprocessing

#### Introduction to NLP Concepts
- **Key Terms**:
  - **Corpus**: Collection of texts used for analysis
  - **Vocabulary**: Set of unique words in a corpus
  - **Documents and Words**: Basic units of text analysis

#### Text Preprocessing Techniques
- **Tokenization**:
  - Splitting text into individual words or sentences using NLTK and SpaCy
- **Stemming and Lemmatization**:
  - **Stemming**: Reducing words to their root form
  - **Lemmatization**: Converting words to their base form
- **Stopword Removal**:
  - Eliminating common words that add little meaning
- **Regex Implementation**:
  - Using regular expressions for advanced text cleaning
- **Parts of Speech (POS) Tagging**:
  - Identifying grammatical parts of text
- **Named Entity Recognition (NER)**:
  - Extracting entities like names, dates, and locations
- **Sentence Boundary Detection**:
  - Identifying sentence beginnings and endings

#### Additional Preprocessing Techniques
- **Lowercasing**:
  - Converting all text to lowercase for uniformity
- **Punctuation Removal**:
  - Stripping punctuation to focus on words
- **Normalization**:
  - Converting text to a consistent format, e.g., numbers or dates
### Text to Vector

#### Text Vectorization Techniques
- **Basic Text Vectorization**:
  - **Bag of Words**: Represents text by word frequency
  - **TF-IDF**: Considers word importance across documents
  - **Word Embeddings**: Dense vector representations

#### Types of Word Embeddings
- **Count or Frequency Based Embeddings**:
  - **One Hot Encoding**: Binary vectors for word presence
  - **Bag of Words**: Simple word count
  - **N-grams**: Sequences of words for context
  - **TF-IDF**: Term frequency-inverse document frequency

- **Deep Learning Based Embeddings**:
  - **Word2Vec**: Captures context using CBOW and Skip-grams
  - **GloVe**: Global vectors for word representation
  - **ELMo**: Contextualized word representations
  - **Doc2Vec**: Extends Word2Vec for entire documents
-----------------------------------------------------------------------------------------------------                          
#### My NLP Journey: --------->>> stage 2
In this stage of my NLP journey, I've explored the following concepts:

## Recurrent Neural Networks (RNN)
- Learned about **hidden states** that help capture sequential data.

## Long Short-Term Memory (LSTM)
- Discovered how LSTMs improve upon RNNs with an added **memory** feature.
- Studied different components:
  - **Cell State**: Maintains long-term dependencies.
  - **Forget Gate**: Decides what information to discard.
  - **Candidate Gate**: Proposes new content for the memory.
  - **Input Gate**: Updates the memory.
  - **Output Gate**: Produces the final output.
- Explored LSTM variants:
  - **Vanilla LSTM**: The standard model.
  - **Stacked LSTM**: Layers for depth.
  - **Bidirectional LSTM**: Processes sequences from both directions.
  - **Peephole LSTM**: Incorporates additional connections.

## Gated Recurrent Unit (GRU)
- Learned about the GRU's efficiency with:
  - **Update Gate**: Manages information retention.
  - **Reset Gate**: Controls new information integration.

## Reflections
- **RNNs** are foundational for sequential tasks but have limitations.
- **LSTMs** enhance memory handling for improved accuracy.
- **GRUs** offer a streamlined alternative without sacrificing performance.                          
-----------------------------------------------------------
#### My NLP Journey: --------->>> stage 3

## Sequence-to-Sequence (seq2seq) Models
- Explored how **encoder-decoder** structures form the basis of seq2seq models.

## Transformers
- **Motivation**: Overcame the limitations of RNNs and LSTMs.
- **Subword Tokenization**:
  - **Byte Pair Encoding (BPE)**
  - **WordPiece**
  - **SentencePiece**
- **Positional Encoding**: Essential for capturing sequence order.

## Transformer Architecture
- **Encoder**:
  - **Attention Mechanism**: Learned about self-attention and multi-head attention.
  - **Layer Normalization**: Improves training stability.
- **Decoder**: Complements the encoder for generating outputs.

### Key Learnings
- Understanding Transformers has paved the way for exploring **Large Language Models (LLMs)**.


---------------------------------------------------
#### My NLP Journey: --------->>> stage 4

## Language Modeling
- Studied **causal language models (CLM)** and **masked language models (MLM)**.
- Learned the differences between **CLM**, **MLM**, and **seq2seq** models.

## BERT (Bidirectional Encoder Representations from Transformers)
- Utilizes masked language modeling (MLM) for:
  - **Next Sentence Prediction**
  - **Question Answering (Q&A)**
  - **Text Classification**
  - **Text Summarization**
  - **Sentiment Analysis**
  - **Named Entity Recognition (NER)**
- Fine-tuning BERT for various NLP tasks.

## Practical Applications
- Used Hugging Face and Tensorflow, Pytorch to implement these models.
- Completed small projects to gain hands-on experience.    
----------------------------------------------------------------------------
#### My NLP Journey: --------->>> stage 5

## LangChain Exploration
- **Model Input/Output**: 
  - Learned about prompts and their types.
  - Explored different models like LLMs and chat models.

## Memory Module
- Studied how memory enhances interactions within LangChain.

## Retrievers
- Investigated various types and their roles in fetching relevant data.

## Chain Types
- **Simple Sequential Chain**
- **Router Chain**
- Understood RAG and its pipeline.

## Advanced LangChain Concepts
- Currently exploring **Agents**, **Tools**, and **Toolkits** for advanced applications.
## Advance Fine tuning Methods 
- Lora/qlora
- Used unsloth  for faster Fine tuning 
- Used Trainer and SFT trainer for Fine Tuning 
                           
                      


                          ''')
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
    elif selected2 == "Time Series":
         markdown_writting("**:blue-background[Click to See  My Overall Time Series Learning journey]**",'''
        # Time Series Analysis: Phase 1 - Statistical Approach

## Understanding Time Series
- **Types of Time Series**: Stationary and non-stationary
- **Stationarity**:
  - Explored **Augmented Dickey-Fuller (ADF) Test** and **Random Walk** concepts
  - Utilized **Autocorrelation Function (ACF)** for analysis

## Time Series Components
- **Trend, Seasonality, and Noise**
- **Decomposition**: Understanding its importance for analysis

## Forecasting Modeling Techniques
- **Exponential Smoothing**:
  - From simple to triple exponential smoothing methods
- **Moving Average Methods**:
  - Applied window statistics for analysis

## Autoregressive Models
- **AR (Autoregressive) Process**
- **ARMA (Autoregressive Moving Average)**
- **ARIMA (AutoRegressive Integrated Moving Average)**:
  - For forecasting non-stationary time series

## Seasonal Models
- **SARIMA (Seasonal ARIMA)**: Incorporates seasonality
- **SARIMAX (Seasonal ARIMA with Exogenous Variables)**:
  - Introduction to multivariate time series

## Vector Models
- **VAR (Vector Autoregression)**: For handling multivariate series
---------------------------------------------------------------------
# Time Series Analysis: Phase 2 - Machine Learning Approach

## Machine Learning Techniques for Time Series
- **Linear Regression**: Applied for basic forecasting tasks.
- **Gradient Boosting**: Utilized to capture complex patterns.
- **XGBoost**: Employed for its efficiency and performance.
- **LightGBM**: Leveraged for handling large datasets and faster training.
------------------------------------------------------------------------
# Time Series Analysis: Phase 3 - Deep Learning Approach

## Deep Learning Techniques for Time Series
- **Recurrent Neural Networks (RNN)**: Implemented for capturing sequential dependencies.
- **Long Short-Term Memory (LSTM)**:
  - Explored various types for enhanced forecasting.
  - Applied to both single-step and multistep multivariate forecasting.
------------------------------------------------------------------------
# Time Series Analysis: Phase 4 - State-of-the-Art Technologies

## Advanced Tools for Time Series
- **Facebook Prophet**: Applied for easy handling of seasonality and holiday effects.
- **Neural Prophet**: Explored for its enhancements over Prophet with neural network capabilities.


                           
                        



                          ''')
         
             
           
             
            
              
      













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

  
