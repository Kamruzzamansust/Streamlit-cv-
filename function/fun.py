import streamlit as st

def project_overview(project_name,Voideo_path,github_link,project_details):    
          with st.container(border = True):
                col1, col2 = st.columns(2,gap = 'large')
                with col1:
                    
                    st.write(project_name)
                    #st.write("----")
                    st.video(Voideo_path)
                    st.write("---")
                    st.markdown(github_link)
                    
                with col2:
                    
                    st.markdown(project_details)
                    
                st.write("----")

def markdown_writting(journey,text):
          with st.container(border = True):
                with st.expander(f"{journey}"):
                 st.markdown(f'''
                    {text}
                ''')
                 

