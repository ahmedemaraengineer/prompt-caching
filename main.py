import os
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM
from langchain_ibm import ChatWatsonx
from llama_index.core import SimpleDirectoryReader
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

set_llm_cache(SQLiteCache(database_path=".langchain.db"))



URL= os.getenv("WATSONX_URL")
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

llm = WatsonxLLM(
    model_id= "ibm/granite-3-8b-instruct", 
    url=URL,
    apikey=WATSONX_APIKEY,
    project_id=WATSONX_PROJECT_ID,
    params={
        GenParams.DECODING_METHOD: "greedy",
        GenParams.TEMPERATURE: 0,
        GenParams.MIN_NEW_TOKENS: 5,
        GenParams.MAX_NEW_TOKENS: 2000,
        GenParams.REPETITION_PENALTY:1.2,
        GenParams.STOP_SEQUENCES: ["\n\n"]
    }
)



now = datetime.now()

prompt = "How are you today?"

resp = llm.invoke(prompt)


end = datetime.now()    

print(f"Consumed {end - now} seconds")
print(resp)