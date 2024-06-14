from langchain.python import PythonREPL
from langchain.agents import Tool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
#from serperAPI import GoogleSerperAPIWrapper
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.utilities import TextRequestsWrapper
from langchain_community.chat_models import AzureChatOpenAI

from langchain_community.callbacks import get_openai_callback
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os, json, re
from bs4 import BeautifulSoup
import requests
from langchain_community.document_loaders import UnstructuredFileLoader,TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import LLMChain

from langchain.prompts import PromptTemplate

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from prompt_tp import fixed_prompt

from google_patent_scraper import scraper_class
import pandas as pd

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Chroma


#向量数据库保存位置
persist_directory = 'docs/chroma/'
 
#创建向量数据库


load_dotenv()


#os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["OPENAI_EMBEDDINGS_DEPLOYMENT"] = "smart-agent-embedding-ada"  # the deployment name for the embedding model
os.environ["OPENAI_EMBEDDINGS_MODEL_NAME"] = "text-embedding-ada-002"
os.environ["OPENAI_API_TYPE"]= os.getenv("OPENAI_API_TYPE")
os.environ["OPENAI_API_VERSION"]= os.getenv("OPENAI_API_VERSION")
os.environ["OPENAI_API_BASE"]= os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_API_KEY"]= os.getenv("OPENAI_API_KEY")



llm = AzureChatOpenAI(
openai_api_base=os.getenv("OPENAI_API_BASE"),
openai_api_version="2023-07-01-preview",
azure_deployment="GPT4-32k",
openai_api_key=os.getenv("OPENAI_API_KEY"),
openai_api_type="azure",
)
#llm = OpenAI(temperature=0)
#
#llm = AzureOpenAI(deployment_id="GPT35", model_name="gpt-3.5-turbo", temperature=0)
#llm = AzureChatOpenAI(deployment_name="GPT35", temperature=0.5)
# embedding = OpenAIEmbeddings(
#                 deployment="your-embeddings-deployment-name",
#                 model="your-embeddings-model-name",
#                 openai_api_base="https://your-endpoint.openai.azure.com/",
#                 openai_api_type="azure",
#             )
#search = GoogleSerperAPIWrapper()

python = PythonREPL()



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)
patent='WO2020006483'
scraper=scraper_class() 

# ~~ Scrape patents individually ~~ #
err_1, soup_1, url_1 = scraper.request_single_patent(patent)
a=scraper.process_patent_html(soup_1)

claims_total=a['claims_total']

# with open(f"code/docs/txt/claims_{patent}.txt", "w") as file:
#     file.write(c)
# # 创建文件对象
# file = open(f'./docs/txt/test_{patent}.txt', "w")

# # 写入数据
# file.write("Hello, World!\n")
# file.write("This is the first line.\n")
# file.write("This is the second line.\n")

# # 关闭文件
# file.close()
# claims_total=a['claims_total']
# loader=TextLoader(claims_total)
# loader = UnstructuredFileLoader(fr"D:\a\program\pythonProject\b.txt")
# # 将文本转成 Document 对象
# loader=TextLoader(f"code/docs/txt/claims_{patent}.txt")
# chain = load_summarize_chain(llm, chain_type="refine", verbose=True)
# documents=loader.load()
# chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

# split_text= text_splitter.split_documents(documents)
# print(split_text)

# vectordb = Chroma.from_documents(
#     documents=split_text,
#     embedding=embedding,
#     persist_directory=persist_directory
# )

# fact_extraction_prompt = PromptTemplate(
#     input_variables=["text_input"],
#     template="{text_input} \n\n please give A comprehensive list of all possible substituents by above claim "
# )

# split_text= text_splitter.split_text(claims_total)
# print(split_text)

# vectordb = Chroma(
#     documents=split_text,
#     embedding=embedding,
#     persist_directory=persist_directory
# )

fact_extraction_prompt = PromptTemplate(
    input_variables=["text_input"],
    template="\n\n\n{text_input} \n\n\n please give A comprehensive list of all possible substituents by above claim  "
)
print(fact_extraction_prompt)
#定义chain
print(claims_total)

fact_extraction_chain = LLMChain(llm=llm, prompt=fact_extraction_prompt)
facts = fact_extraction_chain.run(text_input=claims_total)

print(facts)
