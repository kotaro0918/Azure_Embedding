import openai
import os
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken

df=pd.read_csv(os.path.join(os.getcwd(),'doc_class.csv'))
df_bills=df[['@id','rdfs:label']]
df_bills.columns=["id","label"]
print(df_bills)