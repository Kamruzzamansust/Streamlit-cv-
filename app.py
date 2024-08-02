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
- Learned About Granularity 

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
                col1, col2 = st.columns(2,gap = "large")
                with col1:
                    
                    st.write("1. **Adventure Works Sales Analysis**")
                    #st.write("----")
                    st.video("Power Bi project 2.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/AW-SALES_POWERBI)
                            
                                """)
                    
                with col2:
                    
                    
                    
                     st.markdown("""
       #### Project Description

In this project, I aimed to analyze customer data using DAX and measures. The objectives were as follows:

- **Total Number of Customers:** Calculate the total number of customers.
- **Average Customer Age:** Determine the average age of customers.
- **Age Grouping:** Create a calculated column to group customers by age.
- **Sales Analysis:** Identify that customers aged 31 to 60 are primarily responsible for most sales.
- **Dynamic Top Customers Calculation:** Use parameters to dynamically find top customers based on sales.

This analysis was conducted using DAX, providing insights into customer demographics and sales trends.
        """)
                    
              st.write("----")
              with st.container(border = True):
                col1, col2 = st.columns(2,gap = 'large')
                with col1:
                    
                    st.write("2. **Hospital-Patient Data Analysis** ")
                    #st.write("----")
                    st.video(r"healthpowerbi.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/healthcare-data-analysis-Power-bi)
                            
                                """)
                    
                with col2:
                    
                    st.markdown("""
        #### Project Description

In this project, I aimed to analyze patient trends and generate insights:

- **Monthly Patient Trends:** Analyze patient counts across different months.
- **Weekly Patient Count:** Break down patient visits by week type.
- **Average Waiting Time:** Calculate the average waiting time for patients.
- **Satisfaction Score:** Evaluate patient satisfaction scores.
- **Demographic Analysis:** Explore differences in satisfaction scores and average waiting times among different racesby creating parameters

        
        """)
                    
              st.write("----")
              with st.container(border = True):
                col1, col2 = st.columns(2,gap = 'large')
                with col1:
                    
                    st.write("2. **Transportation Data Analysis** ")
                    #st.write("----")
                    st.video(r"Transportation.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/Transportation-Dashboard-With-Powr-BI)
                            
                                """)
                    
                with col2:
                    
                    st.markdown("""
                                
        #### Project Description                        
This dashboard provides insights into transportation operations using DAX measures. Key features include:

- **Busiest and Least Busy Routes:** Identify the most and least traveled routes.
- **Year-over-Year Profit Analysis:** Examine profit during peak and off-peak hours.
- **Time Group Distribution:** Analyze passenger counts at different times of the day.
- **Bus Utilization:** Assess bus usage efficiency.

To achieve these insights, I used DAX functions such as `CALCULATE()`, `SUMMARIZE()`, `SUMX()`, `TOPN()`, and others.
        """)
                    
              st.write("----")
              with st.container(border = True):
                col1, col2 = st.columns(2,gap = 'large')
                with col1:
                    
                    st.write('4. "DVD RENTAL" Data Base Report Using DAX ')
                    #st.write("----")
                    st.video(r"DVDrental.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/DvD-rental-power-bi)
                            
                                """)
                    
                with col2:
                    
                    st.markdown("""
        
                                #### Project Description

        

In this project, I connected Power BI with a PostgreSQL database to analyze the DVD rental data. Key features of the report include:

- **Top Performing Movies:** Identify movies with the highest performance.
- **Customer Count by Country:** Display the number of customers from various countries.
- **Top Performing Actors:** Highlight actors based on selected movie ratings.
- **Top Customers:** Analyze top customers using DAX calculations.
        """)
    
    elif selected2 == "SQL":
        
        markdown_writting("**:blue-background[Click to See My Overall SQL journey]**","""
                            


# My SQL Journey

## SQL Commands
- **DDL**: Data Definition Language
- **DQL**: Data Query Language
- **DML**: Data Manipulation Language
- **DCL**: Data Control Language
- **TCL**: Transaction Control Language

## Database Concepts
- **Keys**: Primary, Foreign, Composite, Surrogate, Candidate
- **Schema Design**: Data Warehouses, Data Marts, Data Lakes

## SQL Queries
- **Query Execution Order and Clauses**
- **Constraints**
- **Joins**:
  - Inner Join
  - Left and Right Outer Join
  - Full Outer Join
  - Natural Join
  - Self Join

## Subqueries and Advanced Concepts
- **Subqueries**:
  - With FROM
  - With SELECT
  - With WHERE and IN
- **Common Table Expressions (CTEs)**
- **Window Functions**
- **Transactions**: Including GRANT, REVOKE, ROLLBACK

## Indexing
- **Methods**: B-tree, Hash, BRIN, Partial Indexing

## Views
- **Dynamic Views**
- **Materialized Views**

## Stored Procedures
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
                col1, col2 = st.columns(2,gap = 'large')
                with col1:
                    
                    st.write("**1. Food Delivery Status Classification with Xg BOOST and Streamlit** ")
                    #st.write("----")
                    st.video(r"Food.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/Food_Delivery_Status_with_XGBOOST_and_Streamlit)
                            
                                """)
                    
                with col2:
                    
                    st.markdown("""
        ## Project Description

In this project, I developed a classification model using XGBoost to predict food delivery statuses. Key components include:

- **XGBoost Model:** Utilized for accurate classification of delivery statuses.
- **API with FastAPI:** Created an API to request predictions from the model.
- **Streamlit Frontend:** Developed a user-friendly interface for visual interaction with the model via API responses.
- **Docker Compose:** Combined both services to ensure seamless integration and deployment.
        """)
                    
              st.write("----")
              with st.container(border = True):
                 col1, col2 = st.columns(2,gap = 'large')
                 with col1:
                    
                    st.write("**2. Customer Segmentation Ml Classification Problem** ")
                    #st.write("----")
                    st.video(r"customer segmentation.mp4")
                    st.write("---")
                    st.markdown("""
                            [COLAB LINK](https://colab.research.google.com/drive/1ieJWE8-frmn66YyU7hO8PDGFcAAgDg1F#scrollTo=kQ-AxkptE9ik)
                            
                                """)
                    
                 with col2:
                    
                    st.markdown("""
        ##### Project Description

        
This project involves advanced customer segmentation techniques. Key components include:

- **Trade Area Modeling:** Implemented the Huff model to calculate the probability that each customer community will purchase from each store based on distance and attractiveness.
- **Expected Sales and Market Share:** Calculated expected sales and corresponding market share for each store.
- **RFM Analysis:** Ranked customers using Recency, Frequency, and Monetary value.
- **Customer Segmentation:** Applied K-means clustering to segment customers based on RFM values.
        """)
              st.write("----")
              with st.container(border = True):
                 col1, col2 = st.columns(2,gap = 'large')
                 with col1:
                    
                    st.write("**3. Customer Churn Analysis with Logistic Regression** ")
                    #st.write("----")
                    st.video(r"customer_churn_analysis.mp4")
                    st.write("---")
                    st.markdown("""
                            [COLAB LINK](https://colab.research.google.com/drive/1H5zC5xguYU8h44nX3nc29EsrPr7V84jU#scrollTo=B0FRsxgtljw4)
                            
                                """)
                    
                 with col2:
                    
                    st.markdown("""
        ##### Project Description

In this project, I performed customer churn analysis using logistic regression. Key aspects include:

- **Logistic Regression:** Utilized for classifying customer churn.
- **Regularization:** Applied Lasso for regularization to enhance model performance.
- **Hyperparameter Tuning:** Used GridSearchCV to optimize model parameters.
- **Model Accuracy:** Achieved an accuracy of 83%.

        """)
              st.write("----")
              with st.container(border = True):
                        col1, col2 = st.columns(2,gap = 'large')
                        with col1:
                            
                            st.write("**4. Customer Life Time Value Analysis With RFM** ")
                            #st.write("----")
                            st.video(r"CLTV.mp4")
                            st.write("---")
                            st.markdown("""
                                    [COLAB LINK](https://colab.research.google.com/drive/1hUMQL5d3tFAZfdUb30oAmiguLMNlpZei#scrollTo=ZLXaOYw3LM_w)
                                    
                                        """)
                            
                        with col2:
                            
                            st.markdown("""
                ##### Project Description

                
In this project, I focused on analyzing Customer Lifetime Value (CLV) with the following steps:

- **Feature Analysis:** Emphasized important features such as recency, average monetary value, and frequency of buying.
- **RFM Scoring:** Calculated the overall RFM score to evaluate customer value.
- **Clustering:** Used clustering to categorize customers into `low_ltv`, `mid_ltv`, and `high_ltv` segments.
- **Supervised Learning:** Converted the problem into a supervised ML task and applied a Random Forest classifier.
- **Hyperparameter Tuning:** Optimized model parameters using hyperparameter tuning.
- **Model Accuracy:** Achieved an accuracy of 93%..
                """)
                     
              st.write("----")
              with st.container(border = True):
                            col1, col2 = st.columns(2,gap = 'large')
                            with col1:
                                        
                                        st.write("**5. Regression and Clustering problem with pyspark** ")
                                        #st.write("----")
                                        st.video(r"ml_pyspark.mp4")
                                        st.write("---")
                                        st.markdown("""
                                                [DATABRICKS Link](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/4465030434132554/983382902875658/3145564464336035/latest.html)
                                                
                                                    """)
                                        
                            with col2:
                                        
                                        st.markdown("""
                            ##### Project Description

In this Implement a regression model to predict outcomes based on input features.

**Approach**:
1. **Understanding PySpark**:
   - Explored PySpark’s architecture, including the master process, worker process, and cluster manager.
   - Gained familiarity with the PySpark ecosystem components like Spark SQL, Spark Streaming, MLlib, and GraphX.
   
2. **Data Handling and Data Manipulation**:
   - Defined schemas and read CSV files into PySpark DataFrames.
   - Utilized Spark SQL for data selection, filtering, and transformation using `pyspark.sql.functions`.
   -- Applied RDD operations (map, filter, reduce) and User-Defined Functions (UDFs) for initial data processing.

3. **Machine Learning with MLlib**:
   - Implemented regression models and Clustreing using MLlib.
   - Applied featurization, pipelines, transformers, and estimators.
   - Tuned hyperparameters using `CrossValidator` .

5. **Key Parameters**:
   - **Ratings**: Created an RDD of ratings, rows, or tuples.
   - **Rank**: Computed and ranked features.
   - **Lambda**: Applied regularization.
   - **Blocks**: Managed parallel computations with a default value of -1.

                            """)
              st.write("---")
              with st.container(border = True):
                            col1, col2 = st.columns(2,gap = 'large')
                            with col1:
                                        
                                        st.write("**6. Market Busket Analysis with Python Apriori Algorithm** ")
                                        #st.write("----")
                                        st.video(r"mba.mp4")
                                        st.write("---")
                                        st.markdown("""
                                                [COLAB Link](https://colab.research.google.com/drive/17P00B9LTzk3FSD7WfZzaIy9wx_MHkeYu#scrollTo=-SB7jfN1Sd-b)
                                                
                                                    """)
                                        
                            with col2:
                                        
                                        st.markdown("""
                            ##### Project Description

**Objective**: Perform market basket analysis to identify patterns in transactional data and uncover associations between items.

**Approach**:

1. **Understanding Algorithms**:
   - **AIS (Agrawal-Imieli-Smith)**: Explored this early algorithm for association rule mining to understand its principles and limitations.
   - **SETM (Set-oriented Mining)**: Studied this algorithm for efficient mining of frequent itemsets.
   - **FP-Growth (Frequent Pattern Growth)**: Learned about this advanced algorithm that efficiently finds frequent itemsets without candidate generation.

2. **Apriori Algorithm**:
   - **Concept**: Acquired knowledge on the Apriori algorithm, which uses a breadth-first search strategy to find frequent itemsets and generate association rules.
   - **Implementation**:
     - Calculated key metrics to apply the Apriori algorithm:
       - **Support**: Frequency of itemsets appearing in transactions.
       - **Confidence**: Likelihood that an item is purchased when another item is purchased.
       - **Lift**: Ratio of observed support to expected support, indicating the strength of association.

3. **Application**:
   - Applied the Apriori algorithm to transactional data to identify frequent itemsets and generate actionable association rules.
   - Used the calculated metrics (support, confidence, lift) to evaluate and interpret the strength and significance of the discovered associations.
                            """)
              st.write("----")                            
              with st.container(border = True):
                            col1, col2 = st.columns(2,gap = 'large')
                            with col1:
                                        
                                        st.write("**7. Using Apriori Algorithm For Coffe Recommendation** ")
                                        #st.write("----")
                                        st.video(r"apriori.mp4")
                                        st.write("---")
                                        st.markdown("""
                                                [GitHub link](https://github.com/Kamruzzamansust/AprioriAlgorithm-Product-Recommendation-Streamlit-App)
                                                
                                                    """)
                                        
                            with col2:
                                        
                                        st.markdown("""
                            ##### Project Description

   - **Objective**: Utilize the Apriori algorithm to recommend coffee products based on associations found in transactional data.
   - **Application**:
     - Analyzed transactional data to find frequent itemsets related to coffee.
     - Generated association rules to recommend coffee products that are often purchased together.
     - Interpreted the results to provide relevant coffee recommendations based on customer purchase behavior.
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
                    st.markdown("[GitHub link](https://colab.research.google.com/drive/1zDhYWfpiJGptkWrjrzHPrF_ZOK7G8yl7?userstoinvite=raimaadhikary4%40gmail.com&sharingaction=manageaccess&role=writer#scrollTo=q3e9l15TNYk5)")
                    st.write('---')
            with st.container(border = True):
                st.markdown("""
            ##### Project Description

**Objective**: Implement comprehensive workflows for building and training deep learning models using TensorFlow.

**Approach**:

1. **Model Creation**:
   - **Sequential Model**: Built simple linear stacks of layers using the Sequential API.
   - **Functional API**: Created complex architectures with shared layers and branching using the Functional API.
   - **Model Subclassing**: Implemented custom models by subclassing the `tf.keras.Model` class for greater flexibility.

2. **Model Compilation**:
   - Specified loss functions, optimizers, and metrics.
   - Experimented with different configurations to optimize model performance.

3. **Model Training**:
   - Utilized various training techniques, including early stopping and learning rate schedules.
   - Implemented callbacks for monitoring and improving training processes.

4. **Deep Learning Concepts**:
   - Applied concepts like batch normalization, dropout, and advanced optimizers (e.g., Adam, RMSprop) in model creation.
   - Used techniques such as data augmentation to improve model generalization.

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
       
**Objective**: Develop an interactive web app to display available rental properties on a map based on user preferences.

**Approach**:
1. **App Features**:
   - **UI Inputs**: Included controls for users to filter houses based on criteria like price range, number of bedrooms, and location.
   - **Dynamic Map**: Displayed available rental properties on an interactive map.

2. **Interactivity**:
   - **Real-Time Filtering**: Users can adjust filters to update the map view dynamically.
   - **Property Details**: Provided additional information about each property on map markers.

3. **Visualization**:
   - Integrated mapping tools to visualize property locations.
   - Enhanced user experience with a clear and intuitive interface.
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

1. **Dashboard Features**:
   - **Movie Listings**: Users can browse through a list of movies.
   - **Filters**:
     - **Type**: Filter by movies, shows, etc.
     - **Category**: Select specific genres or categories.
     - **Country**: Choose movies based on country of origin.
     - **Release Date**: Narrow down movies by release year or range.

2. **Interactivity**:
   - Real-time updates as users apply different filters.
   - User-friendly interface for seamless navigation.

3. **Visualization**:
   - Displays movie details and statistics.
   - Enhances data exploration with interactive elements.
        """)
              st.write("-----")
              project_overview("Simple Sales Dashboard with R shiny",
                               r"Sales_dashbord_R shiny.mp4",
                               "[GitHub link](https://github.com/Kamruzzamansust/AW-SALES_POWERBI)",
                               """
                               ##### Project Description

1. **Technology Stack**:
   - **R and Shiny**: For building the dashboard.
   - **bs Library**: Utilized for creating a responsive and structured layout.
   - **DT Library**: Used to display interactive tables.
   -use ggplot2 for visualization 

2. **Dashboard Features**:
   - **Visualizations**:
     - Graphs showing total profits month by month.
     - Product sales analysis through bar charts.
   - **Tables**:
     - Interactive tables to display detailed sales data.

3. **Interactivity**:
   - Implemented UI elements for filtering data based on different criteria.
   - Real-time updates as filters are applied.

4. **Visualization**:
   - Enhanced understanding of sales trends and performance.
   - Provided insights into product sales and profitability.


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
                            [GitHub link](https://github.com/Kamruzzamansust/IPL-Dashboard-With-Plotly-Dash)
                            
                                """)
                    
                with col2:
                    
                    st.markdown("""
        ##### Project Description
**Objective**: Develop a multipage dashboard to visualize and analyze various statistics related to IPL matches, including individual performance metrics for batsmen and bowlers.

**Approach**:

1. **Technology Stack**:
   - **Python**: Utilized Python for backend data processing and logic implementation.
   - **Plotly Dash**: Created interactive and dynamic visualizations using Plotly Dash to build the multipage dashboard.

2. **Dashboard Features**:
   - **Multipage Layout**: Designed a dashboard with multiple pages to categorize and present different aspects of IPL data.
   - **Match Statistics**: Visualized various statistics related to IPL matches, such as match outcomes, scores, and team performance.
   - **Batsman Statistics**: Provided detailed statistics on individual batsmen, including runs scored, strike rates, and other performance metrics.
   - **Bowler Statistics**: Showcased individual bowler statistics, including wickets taken, economy rates, and other relevant metrics.

3. **Visualization**:
   - Created interactive charts and graphs to display performance metrics and match statistics.
   - Enabled users to explore and analyze data through various filters and views.
        
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
                            GitHub link--> As file Size is Large can't Push to github
                            
                                """)
                    
                with col2:
                    
                    st.markdown("""
        ##### Project Description

**Objective**: Develop a multipage dashboard to analyze sales data from Adventure Works, providing dynamic insights and interactive exploration.

**Approach**:

1. **Technology Stack**:
   - **Python**: Used for data processing and logic.
   - **Plotly Dash**: Built an interactive and responsive dashboard.

2. **Dashboard Features**:
   - **Multipage Layout**: Organized into multiple pages for different aspects of sales analysis.
   - **Dynamic Insights**: Visualized key sales metrics and trends, such as revenue, product performance, and customer demographics.

3. **Interactivity**:
   - **UI Widgets**: Implemented four interactive widgets to allow users to dynamically change views and filter data.
   - **User Controls**: Enabled selection of time frames, product categories, regions, and other relevant filters.

4. **Visualization**:
   - Created engaging charts and graphs to display sales insights.
   - Provided detailed views and summaries for deeper analysis.
        """)
              st.write("----")
                    
              with st.container(border = True):
                 col1, col2 = st.columns(2)
                 with col1:
                    
                    st.write("2. Youtube Data Visualization with Python Plotly Dash ")
                    #st.write("----")
                    st.video(r"Youtube.mp4")
                    st.write("---")
                    st.markdown("""
                            [GitHub link](https://github.com/Kamruzzamansust/Youtube-analysis-with-dash)
                            
                                """)
                    
                 with col2:
                    
                    st.markdown("""
                                
        #### Project Description

**Objective**: Build a multipage dashboard to analyze YouTube data, highlighting trends and patterns with interactive elements.

**Approach**:

1. **Technology Stack**:
   - **Python and Plotly Dash**: Used for developing the interactive dashboard.

2. **Dashboard Features**:
   - **Multipage Layout with Tabsets**: Organized data into tabbed sections for easy navigation between different analytics areas.
   - **Trend Analysis**: Displayed various YouTube analytics trends, such as views, engagement, and subscriber growth.

3. **Interactivity**:
   - **UI Inputs**: Included dropdowns, sliders, and a date range picker for users to customize data views.
   - **Dynamic Filtering**: Allowed users to explore data based on time periods, categories, and more.

4. **Visualization**:
   - Created detailed visualizations to represent key metrics and trends.
   - Enabled users to gain insights into video performance and audience behavior.
                                
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
**Objective**: Learn and implement text cleaning techniques and word embedding methods to perform sentiment analysis on Twitter data.

**Approach**:

1. **Text Cleaning**:
   - Applied various techniques such as tokenization, stop-word removal, stemming, and lemmatization.
   - Handled special characters, URLs, and emojis specific to Twitter data.
   - Utilized **NLTK** and **spaCy** libraries for advanced text processing tasks.

2. **Word Embedding Techniques**:
   - **Frequency-Based**: Utilized methods like TF-IDF and Count Vectorizer for representing text data.
   - **Deep Learning-Based**: Implemented embeddings like Word2Vec and GloVe for capturing semantic meanings.

3. **Twitter Sentiment Analysis**:
   - Collected and preprocessed Twitter data.
   - Built and trained models to classify sentiment as positive, negative, or neutral.
   - Evaluated model performance using accuracy, precision, and recall metrics.
                                    
           
            """

                                )
                st.write('---')
                project_overview("2. Understanding Transformers ",
                                r"NLP2.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/1DO1H39HiFiFmU1Z-uJaYwlENEpxqoAav#scrollTo=8t1w7p8cu3Al)",
                                """
1. **Core Components**:
   - **Attention Mechanism**: Studied how attention allows the model to focus on relevant parts of the input sequence.
   - **Self-Attention**: Learned how self-attention captures dependencies within a sequence.
   - **Multihead Attention**: Examined how multiple attention heads improve representation learning.

2. **Architecture Details**:
   - **Encoder-Decoder**: Analyzed how the encoder processes inputs and how the decoder generates outputs.
   - **Residual Connections**: Understood their role in improving gradient flow and network stability.
   - **Layer Normalization**: Investigated how it stabilizes training and accelerates convergence.

3. **Additional Components**:
   - **Softmax**: Learned its use in transforming attention scores into probabilities.
   - **Feedforward Neural Network**: Explored its role in processing the output of the attention mechanism.

4. **Decoding Process**:
   - Studied how the decoder creates a context vector and uses it to generate sequences.
                                    
           
            """)
                
                st.write('---')
                project_overview("3. Architecture Development Of Transformers",
                                 r"NLP3.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/1dLbRS5BPRyswrhkv0wv7WcvFgcPRC3hP#scrollTo=fnnfAaKCIcDt)",
                                """
                                   **Objective**: Develop a detailed, object-oriented representation of the Transformer architecture, incorporating its key components to understand its inner workings better.

**Approach**:

1. **Core Components**:
   - **Self-Attention Mechanism**: Created a class to model the self-attention mechanism, which includes splitting heads, computing attention scores, and applying the attention weights.
   - **Multihead Attention**: Implemented as a separate class to manage multiple self-attention heads, combining their outputs and applying a final linear transformation.
   - **Feedforward Neural Network**: Modeled as a class to handle position-wise feedforward operations after the attention mechanism.
   - **Residual Connections**: Incorporated as part of the layer classes to add input to the output of the transformations, enhancing gradient flow and stability.
   - **Layer Normalization**: Added normalization layers to stabilize training by normalizing the inputs of each layer.

2. **Architecture Details**:
   - **Encoder**: Developed an encoder class consisting of multiple layers of multihead attention followed by feedforward neural networks, each with residual connections and layer normalization.
   - **Decoder**: Designed a decoder class with similar layers as the encoder but with additional cross-attention layers to incorporate encoder outputs, enabling the generation of sequences.

3. **Additional Features**:
   - **Positional Encoding**: Incorporated positional encoding to add sequence information to the embeddings.
   - **Embedding Layer**: Created an embedding class to transform input tokens into dense vectors.

4. **Implementation**:
   - Used Object-Oriented Programming principles to design modular and reusable components. 
           
            """)
        with st.expander("**NLP INtermediate**"):
              st.write('---')
              project_overview("1. **Trying To Know about BERT**",
                                 r"BERT1.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/1edixmSMEUSPj6eO5V1_xLExkYwgeqQIr#scrollTo=qQDChvozM1QJ)",
                                """
**Objective**: Gain a deep understanding of various types of LLMs, their training objectives, and specific implementations like BERT and its variants.

**Approach**:

1. **Types of LLMs Based on Training Objectives**:
   - **Autoregressive Models**: Models that predict the next token in a sequence given the previous tokens. Example: GPT series.
   - **Autoencoding Models**: Models that learn to reconstruct the input from a corrupted version of it. Example: BERT.
   - **Sequence-to-Sequence (Seq2Seq) Models**: Models designed to transform one sequence into another. Example: T5.
   - **Hybrid Models**: Models that combine aspects of autoregressive and autoencoding objectives. Example: T5, which can perform both generation and understanding tasks.

2. **BERT Workflow**:
   - **Masked Language Modeling (MLM)**: Understood how BERT predicts masked tokens in a sentence.
   - **Next Sentence Prediction (NSP)**: Studied how BERT learns to predict whether one sentence follows another.
   - **Special Tokens**: Explored the role of tokens like `[CLS]` for classification and `[SEP]` for separating segments.

3. **Utilizing Transformers Library**:
   - **Pipelines**: Learned to use the Transformers library’s pipeline for various tasks like text classification, named entity recognition, and question answering.
   - **Loading Pretrained Models**: Practiced loading and fine-tuning pretrained models from the Transformers library.

4. **Exploring BERT Variants**:
   - **RoBERTa**: Investigated improvements and differences from BERT, focusing on training without NSP.
   - **ALBERT**: Studied the parameter reduction techniques and their impact on model performance and efficiency.
   - ElECTRA , DistilBERT and more 

**Outcome**:
This notebook provided a thorough theoretical understanding of LLMs, focusing on the mechanics of BERT and its variants, and practical experience with the Transformers library. It facilitated an in-depth exploration of model types, objectives, and advanced variations.
                                    
           
            """)
              st.write("---")
              project_overview("2. **Masked Language modeling and Next Sentence Prediction With BERT** ",
                                 r"BERTMASKED.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/18lG5UvQ8N1djCeHDwXV9d6zEUhnYgoyQ#scrollTo=qdHfE77rCBLs)",
                                """
                                  **Objective**: Gain a comprehensive understanding of masked modeling with BERT, including its pretraining tasks, application of various techniques, and comparison with other models.

**Approach**:

1. **Masked Modeling in BERT**:
   - **Understanding BERT's Tasks**: Explored BERT's pretraining tasks, including Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).
   - **BERT Tokenizer**: Utilized the BERT tokenizer from the Transformers library to preprocess text data.
   - **Attention Mask and Mask Index**: Learned how attention masks and mask indices work in the context of BERT's training and prediction.

2. **Model Application**:
   - **Logits and Softmax**: Analyzed the model's output logits and applied the softmax function to obtain probabilities.
   - **Top-K**: Understood how to extract the top-k predictions from the model's output.
   - **Pretrained Model**: Applied the pretrained model `deepset/bert-base-cased-squad2` for specific tasks like question answering.

3. **Advanced Techniques**:
   - **Cosine Similarity**: Implemented cosine similarity to measure semantic similarity between text embeddings.
   - **Semantic Search**: Explored techniques for semantic search using BERT embeddings.

4. **Additional Models and Methods**:
   - **Sentence Transformers**: Utilized sentence transformers to encode sentences and improve similarity measures.
   - **Text Generation with GPT-2**: Investigated text generation capabilities using GPT-2, including methods like greedy search and beam search.

**Outcome**:
This notebook provided an in-depth exploration of masked modeling with BERT, covering key concepts and techniques. By using the Transformers library, pretrained models, and additional methods, the project enhanced understanding of BERT's functionality and compared it with other models like GPT-2 for various NLP tasks.  
           
            """)
              st.write("---")
              project_overview("3. **BERT Fine Tuning ( Without Pipeline )** ",
                                 r"BERT2.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/1rM0prvLLajG81bZew8bhykjTtiQ2pqEU#scrollTo=iravzBml_Dvu)",
                                """
**Approach**:
                                
1. **Setup and Initialization**:
   - **Transformers Library**: Utilized the `transformers` library for working with BERT.
   - **BERT Tokenizer and Model**: Initialized the BERT tokenizer and loaded the `bert-base-cased` model.

2. **Data Preparation**:
   - **Load Data**: Imported the dataset using Pandas.
   - **Custom Dataset**: Defined a custom dataset class for PyTorch to handle the tokenization and formatting of text data.
   - **Data Loaders**: Created data loaders for training and validation, splitting the dataset accordingly.
   - **Constants**: Set constants such as `MAX_LEN` (maximum sequence length) and `BATCH_SIZE`.

3. **Model Training Setup**:
   - **Model Loading**: Loaded the BERT model with a classification head.
   - **Optimizers and Schedulers**: Defined optimizers and learning rate schedulers for training.
   - **Loss Function**: Set up the loss function for classification.

4. **Training Process**:
   - **Training Functions**: Created functions for training the model, including forward pass, loss computation, and backpropagation.
   - **Evaluation Function**: Developed a function to evaluate model performance on the validation set.
   - **Training Loop**: Implemented the training loop to iterate over epochs, perform training and validation, and track metrics.

5. **Post-Training Analysis**:
   - **Plot Metrics**: Visualized training and validation metrics to assess model performance.
   - **Prediction Function**: Created a function to make predictions with the fine-tuned model.
   - **Confusion Matrix**: Generated and displayed a confusion matrix to evaluate classification performance.

6. **Model Saving and Loading**:
   - **Save Model**: Saved the fine-tuned model for future use.
   - **Load Model**: Reloaded the model for making predictions on new data.  
           
            """)
              st.write("---")
              project_overview("4. **Fine Tuning DistilBERT With Custom Datest** ",
                                 r"BERT3.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/1qvcFj94l4-IrQ17X2tkoMwdE4RV8e96l#scrollTo=3uJ90BUf9gU1)",
                                """
           **Objective**: Fine-tune the DistilBERT model for text classification using BBC text data from Kaggle.

**Approach**:

1. **Setup and Initialization**:
   - **Libraries**: Utilized the `transformers`, `torch`, and `accelerate` libraries for model training and evaluation.
   - **Dataset**: Loaded the BBC text dataset from Kaggle.

2. **Data Preparation**:
   - **Data Splitting**: Divided the dataset into training and test sets.
   - **Tokenization**: Tokenized the text data using the DistilBERT tokenizer.

3. **Model Definition**:
   - **DistilBERT**: Used the `distilbert-base-uncased` model for text classification.
   - **Data Class**: Created a custom data class to handle the tokenized data.

4. **Training Setup**:
   - **TrainingArguments**: Configured training parameters using `TrainingArguments()`.
   - **Trainer**: Employed the `Trainer()` class to manage the training process.

5. **Training and Evaluation**:
   - **Training**: Fine-tuned the DistilBERT model on the training set.
   - **Evaluation**: Evaluated the model’s performance on the test set.

6. **Model Management**:
   - **Saving Model**: Saved the fine-tuned model for future use.
   - **Loading and Prediction**: Loaded the saved model and made predictions on new data.
                         
           
            """)
              st.write("---")
              project_overview("5. **Integration of Fine-Tuned DistilBERT Model with FastAPI and R Shiny** ",
                                 r"finetune.mp4",
                                "[GITHUB Link](https://github.com/Kamruzzamansust/Fine-tune-DistilBERT-for-Text-Classification)",
                                """


**Objective**: Develop an interactive web application that utilizes the fine-tuned DistilBERT text classification model through a FastAPI backend and an R Shiny frontend.

**Approach**:

1. **API Development**:
   - **FastAPI**: Created an API using FastAPI to serve the fine-tuned DistilBERT model for text classification. This API handles text input, performs classification, and returns the results.

2. **Web Application Development**:
   - **R Shiny UI**: Designed a user interface with R Shiny to allow users to input text and view classification results.
   - **API Integration**: Utilized the `httr` library in R to make requests to the FastAPI endpoint from the Shiny app. This integration enables real-time interaction with the model.

3. **Workflow**:
   - **Text Input**: Users input text into the Shiny web app.
   - **API Call**: The Shiny app sends the text to the FastAPI endpoint.
   - **Model Prediction**: The FastAPI backend uses the fine-tuned DistilBERT model to classify the text.
   - **Display Results**: The classification results are returned to the Shiny app and displayed to the user.                        
           
            """)
              st.write("---")
              project_overview("6. **Trying To understand QA system** ",
                                 r"QA.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/1LrWV8ugDtLqozeNmjnVWnjGgraAG3eq0#scrollTo=EhCpzkFcb0mW)",
                                """
**Objective**: Understand open-domain and closed-domain question answering and implement a QA system using BERT.

**Approach**:

1. **Domain Understanding**:
   - **Open-Domain**: Answer questions from a large, unstructured dataset.
   - **Closed-Domain**: Answer questions from a specific, predefined set of documents.

2. **QA with BERT**:
   - **Model and Tokenizer**: Imported `BertForQuestionAnswering` from the `transformers` library.
   - **Initialization**: Used `bert-large-uncased-whole-word-masking-finetuned-squad` model.
   - **Example Setup**:
     - Created a sample question.
     - Provided reference text from which the answer will be extracted.

3. **Process**:
   - **Tokenization**: Performed tokenization on the input text.
   - **Indexing**: Found token indices for question and context.
   - **Segment Embeddings**: Created segment embeddings to differentiate between question and context.
   - **Model Scoring**: Obtained model scores for start and end positions.
   - **Extract Answer**: Identified and returned the answer based on scores.                        
           
            """)
              st.write("---")
              project_overview("7. Text Summarization using Transfer Learning With GPT-2, google-pegasus,T5",
                                 r"Text_Summarization_simple.mp4",
                                "[COLAB link](https://colab.research.google.com/drive/1WNG2ZVvSBpuK3ZBEFCm8b1SV1Y1os1NY#scrollTo=LFEy_b0FU4U4)",
                                """
**Objective**: Implement text summarization using advanced models and evaluate with BLEU metric.

**Models Used**:
- **GPT-2**
- **Google PEGASUS**
- **T5**

**Approach**:
1. **Model Selection**: Utilized transfer learning with pre-trained models for effective summarization.
2. **Implementation**: 
   - Fine-tuned models on specific datasets.
   - Generated summaries for given text inputs.
3. **Evaluation**:
   - Calculated the BLEU metric to assess the quality of generated summaries.   
           
            """)
              st.write("---")
              project_overview("8. **Question Answering With BERT** ",
                                 r"Bert_qa_model.mp4",
                                "[GTHUB Link](https://github.com/Kamruzzamansust/Q_A-with-BERT)",
                                """


**Objective**: Develop a user-friendly application for question answering using BERT.

**Features**:
- **User Input**:
  - Text area for reference text input.
  - Text input for user questions.
  
- **Functionality**:
  - Utilizes BERT model to extract answers from the provided text.
  - Displays the answer based on user queries.
                        
           
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

  
