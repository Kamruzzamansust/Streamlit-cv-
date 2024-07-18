import streamlit as st
from  streamlit_option_menu import  option_menu 
from streamlit_lottie import st_lottie
import json
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
    selected2 = option_menu(
            menu_title = None,
            options =['Power Bi','SQL','ML',"DASH",'R-Shiny',"Tensorflow","NLP","Time Series"],
            orientation= 'horizontal'
    )
    
    if selected2 == "Power Bi":
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
    elif selected2 == "ML":
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
                                
                                st.write("**4. Regression and Clustering problem with pyspark** ")
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
                    
                    st.write("**2. Netdlix Data Visualisation R-Shiny** ")
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

  
