import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

os.getenv("GROQ_API_KEY")

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            model_name="llama-3.1-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0)
        
    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### Extracted content from the website:
            {page_content}
            ### Task:
            This text is sourced from the careers section of the website.
            Your objective is to identify the job listings and present them in JSON format with the keys: 'role', 'experience', 'skills', and 'description'. Please return only the valid JSON.
            ### Expected JSON Output (without any introduction)
            """
)

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            (
            """
            ### POSITION DETAILS:
            {job_description}
            
            ### TASK:
            You are Elon Musk, a business development executive at TCS, a company specializing in AI and software solutions. TCS focuses on helping businesses streamline their processes through automation. 
            Your role is to draft a cold email to a potential client regarding the position mentioned above, emphasizing TCS's capabilities in addressing their needs.
            Additionally, incorporate key highlights from the following links to showcase TCS's portfolio: {link_list}
            Remember to maintain your identity as Elon Musk, BDE at TCS. 
            Do not include any introductory remarks.
            ### EMAIL CONTENT (NO INTRODUCTION):
            """
            )
    )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content