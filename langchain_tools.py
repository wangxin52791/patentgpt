from langchain.python import PythonREPL
from langchain.agents import Tool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
#from serperAPI import GoogleSerperAPIWrapper
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.utilities import TextRequestsWrapper
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.llms import AzureOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os, json, re
from bs4 import BeautifulSoup
import requests
from langchain.chains import LLMChain
from langchain.document_loaders import UnstructuredFileLoader,TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts import PromptTemplate

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from prompt_tp import fixed_prompt

from google_patent_scraper import scraper_class
import pandas as pd
import json
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma

load_dotenv()
embedding = OpenAIEmbeddings()

#os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

os.environ["OPENAI_API_TYPE"]= os.getenv("OPENAI_API_TYPE")
os.environ["OPENAI_API_VERSION"]= os.getenv("OPENAI_API_VERSION")
os.environ["OPENAI_API_BASE"]= os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_API_KEY"]= os.getenv("OPENAI_API_KEY")

#llm = OpenAI(temperature=0)
llm = AzureChatOpenAI(deployment_name="GPT4-8k", temperature=0)
#llm = AzureOpenAI(deployment_id="GPT35", model_name="gpt-3.5-turbo", temperature=0)
#llm = AzureChatOpenAI(deployment_name="GPT35", temperature=0.5)

#search = GoogleSerperAPIWrapper()
python = PythonREPL()

#向量数据库保存位置
persist_directory = 'code/docs/chroma/'

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)
def Claims_summary(patent):
    try:
        scraper=scraper_class() 

        # ~~ Scrape patents individually ~~ #
        err_1, soup_1, url_1 = scraper.request_single_patent(patent)
        a=scraper.process_patent_html(soup_1)

        c=a['claims_total']
        with open(f"code/docs/txt/claims_{patent}.txt", "w") as file:
            file.write(c)
  
        loader=TextLoader(f"code/docs/txt/claims_{patent}.txt")
        # loader = UnstructuredFileLoader(fr"D:\a\program\pythonProject\b.txt")
        # # 将文本转成 Document 对象
        documents=loader.load()
        # chain = load_summarize_chain(llm, chain_type="refine", verbose=True)
        
        split_text= text_splitter.split_documents(documents)
        # print(split_text)

        # vectordb = Chroma.from_documents(
        #     documents=split_text,
        #     embedding=embedding,
        #     persist_directory=persist_directory
        # )

        fact_extraction_prompt = PromptTemplate(
            input_variables=["text_input"],
            template="{text_input} \n\n please give A comprehensive list of all possible substituents by above claim "
        )
        
        #定义chain
        fact_extraction_chain = LLMChain(llm=llm, prompt=fact_extraction_prompt)
        facts = fact_extraction_chain.run(c)
        return facts
    except:
        return 'pantentid wrong'
    
def calculate_properties(smiles):
    try:
        smiles = smiles.split(".")[0]
        mol = Chem.MolFromSmiles(str(smiles))

        return str({
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "AcceptorH": Descriptors.NumHAcceptors(mol),
            "DonorsH": Descriptors.NumHDonors(mol)
        })
    except:
        return "SMILES missing, calculate_properties is not valid for the wrong SMILES, try search the SMILES first and then do the calculation of properties again"

def calculate_properties_byname(drugname):


    smiles = search_SMILES(drugname).split(":")[-1].strip()
    smiles = smiles.split(".")[0]
    mol = Chem.MolFromSmiles(str(smiles))

    return str({
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "AcceptorH": Descriptors.NumHAcceptors(mol),
        "DonorsH": Descriptors.NumHDonors(mol)
    })

def calculate_similarity(smiles_str):
    try:
        smiles_list = smiles_str.split(",")
        mol1 = Chem.MolFromSmiles(smiles_list[0])
        mol2 = Chem.MolFromSmiles(smiles_list[1])

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)

        # Calculate Tanimoto similarity between the two fingerprints
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

        return str(similarity)

    except:
        return "SMILES missing, calculate_properties is not valid for the wrong SMILES, try search the SMILES first and then do the calculation of similarity again"


def search_SMILES(drugname):
    try:
        # pubchem API to get SMILES 
        requests = TextRequestsWrapper()
        response = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drugname}/property/CanonicalSMILES/json")

        response = json.loads(response)
        smiles = response["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        return f"{drugname} SMILES: {smiles}"
    except:
        return "There are something wrong with input SIMLES, try to split them first? or we cannot find"

def search_drug_image(drugname):
    import base64

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drugname}/PNG"

    return str({"img": url})

def get_download_link(patent):
    headers = {
            'user-agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.55 Safari/537.36',
            'Content-Type': 'text/html; charset=utf-8'}
    url = f"https://patents.google.com/patent/{patent}/en?oq={patent}"
    response = requests.get(url, headers=headers)

    html_content = response.text
    links = re.findall(r"https:\/\/patentimages.storage.googleapis.com\/.*\.pdf", html_content)

    return links[0]

def Read_pdf(patent_address):
    from PyPDF2 import PdfReader
    import io
    from PIL import Image
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Windows; Windows x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36'}

    url = patent_address
    response = requests.get(url=url, headers=headers, verify=False)
    on_fly_mem_obj = io.BytesIO(response.content)
    pdf_file = PdfReader(on_fly_mem_obj)

    pdf_page = pdf_file.getPage(0)

    # Convert the PDF page to a PNG image
    img = pdf_page.to_png()

    pil_img = Image.open(io.BytesIO(img))

    # Save the PIL Image object as a file
    pil_img.save('example.png')


class WebPageTool(BaseTool):
    name = "Get Webpage"
    description = "Useful for when you need to get the content from a specific webpage"

    def _run(self, webpage: str):
        headers = {'Content-Type': 'text/html; charset=utf-8'}
        response = requests.get(webpage, headers=headers)

        html_content = response.text

        def strip_html_tags(html_content):
            soup = BeautifulSoup(html_content, "html.parser")
            stripped_text = soup.get_text()
            return stripped_text

        stripped_content = strip_html_tags(html_content)
        if len(stripped_content) > 4000:
            stripped_content = stripped_content[:4000]
        return stripped_content
    
    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")
    
class WebPatentTool(BaseTool):
    name = "Get Patent Webpage"
    description = "Useful for when you need to get the patent content from https://patents.google.com/. Input should be like WO2020006483"

    def _run(self, webpage: str):
        headers = {
            'user-agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.55 Safari/537.36',
            'Content-Type': 'text/html; charset=utf-8'}
        url = f"https://patents.google.com/patent/{webpage}/en?oq={webpage}"
        response = requests.get(url, headers=headers)

        html_content = response.text

        def strip_html_tags(html_content):
            soup = BeautifulSoup(html_content, "html.parser")
            stripped_text = soup.get_text()
            return stripped_text

        stripped_content = strip_html_tags(html_content)
        if len(stripped_content) > 4000:
            stripped_content = stripped_content[:4000]
        return stripped_content
    
    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")
    

page_getter = WebPageTool()
patent_getter = WebPatentTool()

tools = [
    # Tool(
    #     name = "Current Search",
    #     #func=search.run,
    #     description="useful for when you need to answer questions about today/current information , today/current events or the today/current state of the world",
        
    # ),
    Tool(
        name = "Search drug SMILES",
        func=lambda s: search_SMILES(s),
        description="useful for when you need to search the SMILES by drugname. Input is a single drugname, Output is the SMILES string. \
            If the input parameters contains comma, split them firstly and then search them one by one",
        # return_direct=True
    ),
    Tool(
        name = "Search drug image",
        func=lambda s: search_drug_image(s),
        description="useful for when you need to search the structure images by drugname.",
        return_direct=True
    ),
    Tool(
        name="Calculate properties by SMILES",
        func=lambda s: calculate_properties(s),
        description="useful when you want to calculate the molecular properties. \
        Input: Should be molecular SMILES. eg. O=C(Oc1ccccc1C(=O)O)C \
        If you receive two and more compounds with `and` words, first you should split it \
        If you receive drug words eg. Abemaciclib, you should do `Action: Search drug` to get the SMILES first",
    ),
    Tool(
        name="Calculate properties by drugname",
        func=lambda s: calculate_properties_byname(s),
        description="useful when you want to calculate the molecular properties. \
        Input: Should be molecular name. eg. Abemaciclib \
        If you receive two and more compounds with `and` words, first you should split it ",
    ),
    Tool(
        name="Calculate similarity",
        func=lambda s: calculate_similarity(s),
        description="useful when you want to calculate the two compound similarity score. Input should be a list. Firstly split the two compounds into the list like 'CC1,CCC' ",
        # return_direct=True
    ),
    page_getter,
    patent_getter,
    Tool(
        name="Get patent_download link",
        func=lambda s: get_download_link(s),
        description="useful when you want to get the download pdf link for a patent, input should be a patent id",
        # return_direct=True
    ),
     Tool(
         name="summary the patent claims",
         func=lambda s: Claims_summary(s),
         description="use for patent claims summmary ,you can get credical structure",
     )
    
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
    verbose=True, 
    memory=memory,
    early_stopping_method='generate'
    )

agent.agent.llm_chain.prompt.messages[0].prompt.template = fixed_prompt


def ag_plugin(questions):
    with get_openai_callback() as cb:
        try:
            response = agent.run(questions)
        except ValueError as e:
            response = str(e)

        response = json.dumps(response)
        response_msg = {
            'usage': {'prompt_tokens': cb.prompt_tokens, 'completion_tokens': cb.completion_tokens, 'total_tokens': cb.total_tokens},
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': response 
                    },
                    'finish_reason': 'stop',
                    'index': 0
                }
            ]
        }

        print(response_msg)
    return json.dumps(response_msg)

if __name__ == "__main__":
    
    #ag_plugin("give me some approvded drugs for KRAS target")
    #ag_plugin("Get the SMILES of Abemaciclib")
    #ag_plugin("What are the titles of the top stories on www.cbsnews.com/?")
    #ag_plugin("Could you get the clinical informations about Abemaciclib on https://pubchem.ncbi.nlm.nih.gov/")

    #ag_plugin("Calculate the molecular properties of the Abemaciclib")
    #Get the structure image of Abemaciclib
    #transform above properties sentence into markdown table
    #ag_plugin("Get the SMILES of Abrocitinib")
    # ag_plugin("Calculate the similarity between Abemaciclib and Abrocitinib")
    # tell me some informations of patent WO2020006483
    # Give me the download pdf link of WO2020006483
    # tell me some information of patent, which id is WO2020006483
    # Tell me some information about IC50 included in this patent
    ag_plugin("summary patent WO2020006483 claims")
    # ag_plugin("What's the date today?")
