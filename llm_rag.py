import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

# Load the data
print("Loading and formatting data...\n")
df = pd.read_csv('../data/IMDB.csv')

# drop rows not in type map, this is for simplicity
typeMap = {'movie': 'movie',
           'tvMovie': 'movie',
           'tvSeries': 'TV show',
           'tvMiniSeries': 'mini series'}
df = df[df.titleType.isin(typeMap.keys())]
df = df.reset_index(drop=True)

# create a duplicate titleType column with better names for creating a prompt
df['media_type'] = df.titleType.apply(lambda x: typeMap[x])

# space out the genres column for better readability
df['genres_spaced'] = df.genres.apply(lambda x: x.replace(',', ', '))

# create a full description column to cast into vector space for rag
# note! There is more information in the dataset we could be using to improve this further!
df['full_description'] = df.apply(lambda x: f"'{x.primaryTitle}' is a {x.genres_spaced} \
	                                          {x.media_type}. The description is: {x.Description}"
	                                          ,axis=1)

# split the text and cast it into a numberical embedding.
print("Embedding text...\n")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.create_documents(df['full_description'])
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Set up for the LLM
print("Initializing LLM...\n")
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# create chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#Test cases
print("Performing test cases...\n")
queries = ["Can you suggest a light hearted tv show?",
           "Can you suggest a horror movie?",
           "Can you suggest a movie similar to The Babadook?",
           "Why is the sky blue?"]

for query in queries:
	print(f"query: {query}")
	print(f"response: {rag_chain.invoke(query)}\n\n")