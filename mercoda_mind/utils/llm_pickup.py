from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


def pick_llm(level : str):

    if level.lower() == 'low':
        llm = ChatOpenAI(model='gpt-5.6-luna', temperature=0.5)
    elif level.lower() == 'medium':
        llm = ChatOpenAI(model='gpt-5.6-terra', temperature=0.5)
    elif level.lower() == 'high':
        llm = ChatOpenAI(model='gpt-5.6-sol', temperature=0.5)
    else:
        raise ValueError(f'unsupported level {level}')

    return llm

if __name__ == '__main__':
    obj_llm = pick_llm('low')
    print(obj_llm.invoke('what is capital of England').content)