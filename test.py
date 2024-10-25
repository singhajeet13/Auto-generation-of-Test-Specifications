import streamlit as st
import PyPDF2

from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain_core.prompts import PromptTemplate

# Initialize Ollama/LLM model
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI


import pandas as pd
import re

#############################################
# Setting up streamlit app and input from user (UI)
##################################################


st.title("LLM PDF and Text Processor")
with st.sidebar:
    temperature  = st.slider("temperature ", 0.0, 1.0, 0.1)


# Input section
input_text = st.text_area("Enter additional information (Text format):")

uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])



#####################################################
# DEFINING LLM Model
##################################################

llm = ChatOpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
    model = "llama3:latest",
    temperature = temperature
)

# llm = OllamaLLM(model="llama3:latest",
#                 temperature = temperature)



########################################################
# PROCESSING THE PDF AND GENERATING RESPONSE
########################################################

if uploaded_pdf:
    st.write(input_text)

    #Read PDF file
    reader = PyPDF2.PdfReader(uploaded_pdf)

    ## Final dataframe - after merging the output from each page 
    final_df = pd.DataFrame(columns=["Component of Transformer", "Benefit"])

    ## Loop through each page

    for page_num in range(len(reader.pages)):

        #parse text from one page
        pdf_text = reader.pages[page_num].extract_text()

        # Combine user input text with extracted PDF text
        combined_text = input_text + "\n" + pdf_text

        # Define the prompt for the LLM
        prompt_template = PromptTemplate(
            input_variables=["text"],
            template="""
            You are a research scholor working in Airtificial Intelligence domain, and you are specialised for Transformer used in NLP tasks.
            
            Text: {text}
            """
        )

        

        # Create an LLMChain with the prompt and parser
        llm_chain = LLMChain(prompt=prompt_template, llm=llm)

        # Run the LLM chain with the combined input
        response = llm_chain.run(combined_text)

        #Convert respose into datafram
        # Use regular expression to find all rows of the table // 
        # THIS IS SPECIFIC TO DUMMY USE CASE (for llama-3 model) - Need to update based on output format from model 
        pattern = r"\|\s*(.*?)\s*\|\s*(.*?)\s*\|"
        matches = re.findall(pattern, response)

        # Filter out the header and separator lines if they are captured
        data = [match for match in matches if "---" not in match[0]]

        # Create DataFrame, and fill the respose from single/current page
        df = pd.DataFrame(data, columns=["Component of Transformer", "Benefit"])

        # add the current page output to final data frame
        final_df = pd.concat([final_df, df], axis = 0)

        st.write("Data from Page - {}".format(page_num))
        st.write(response)
        
    

    st.dataframe(final_df)
    print(final_df)
