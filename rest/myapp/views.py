from django.shortcuts import render
from django.conf import settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
from django.http import HttpResponse, HttpRequest , HttpResponseNotFound
import os
import getpass
from langchain_community.llms.ollama import Ollama
from langchain_core.tools import tool
import pandas as pd
from langchain_community.document_loaders import YoutubeLoader
from langgraph.graph import StateGraph, END, START
from typing import Annotated, TypedDict
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
import ssl
import random
ssl._create_default_https_context = ssl._create_stdlib_context

from io import BytesIO
from elevenlabs import VoiceSettings, play, save
from elevenlabs.client import ElevenLabs
import ffmpeg
import IPython.display as ipd
import weave
import re

class AgentState(TypedDict):
    input: str
    transcript: str
    title: str
    summarized_text: str
    translate_op: int
    translated_str: str
    output_audio: str | None = None
    rec: tuple
    thumbnail_url: str
    safety: str


os.environ["OPENAI_API_KEY"] = (
    ""
)

os.environ["PINECONE_API_KEY"] = ""
from langchain_together import Together

guard = Together(
    model="meta-llama/LlamaGuard-2-8b",
    together_api_key="",
)


def safety_check(state: AgentState) -> AgentState:
    ### Llama Guard

    guard_temp_ans = """
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes. 
S2: Non-Violent Crimes. 
S3: Sex Crimes. 
S4: Child Exploitation. 
S5: Defamation. 
S6: Specialized Advice. 
S8: Intellectual Property. 
S9: Indiscriminate Weapons. 
S10: Hate. 
S11: Self-Harm. 
S12: Sexual Content. 
S13: Elections. 
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: {answer}

<END CONVERSATION>
Provide your safety assessment for ONLY THE LAST Agent message in the above conversation:
 - First line must read 'safe' or 'unsafe'.
 - If unsafe, a second line must include a comma-separated list of violated categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    prompt_ans_guard = PromptTemplate.from_template(guard_temp_ans)

    guard_chain = prompt_ans_guard | guard
    cuted_text=state['transcript'].split()
    cuted_text= cuted_text[:6000]
    cuted_text=' '.join(cuted_text)
    safety = guard_chain.invoke({"answer":cuted_text})

    if "unsafe" in safety:
        print("no safe ", safety)
        state['safety']='0'
        print("no safe ", state['safety'])

        return "0"
    else:
        print(safety)
        state['safety']='1'
        return "1"


def loader(state: AgentState) -> AgentState:
    video_script = YoutubeLoader.from_youtube_url(
        state["input"], add_video_info=True, language=["en", "ar" , 'en-US']
    )
    video_script = video_script.load()

    transcript = video_script[0].page_content
    title = video_script[0].metadata["title"]
    thumbnail_url = video_script[0].metadata["thumbnail_url"]

    return {
        "transcript": transcript,
        "input": state["input"],
        "title": title,
        "thumbnail_url": thumbnail_url,
    }


def summary(state: AgentState) -> AgentState:
    state['safety']='1'
    print(state["title"])
    llm = Ollama(model="llama3.1")
    prompt = PromptTemplate.from_template(
        """
    You are an agent that will summarize the text and try to make the best summary and make it as bulltes points . your answer should start immediately with the summary 

    transcript : {transcript}

  

    Your answer :
    """
    )

    chain = prompt | llm
    summarized_text = chain.invoke({"transcript": state["transcript"]})
    state["summarized_text"] = summarized_text
    print(summarized_text)
    return state


def translate(state: AgentState) -> AgentState:
    print("in translate ")
    print(state["translate_op"])
    if state["translate_op"] == "1":

        promt = PromptTemplate.from_template(
            """Transalte this text into arabic .
                                        text:{text}
                                        
                                        """
        )
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # chat = ChatWatsonx(
        #     model_id="sdaia/allam-1-13b-instruct",
        #     url="https://eu-de.ml.cloud.ibm.com",
        #     project_id="96ef88c6-4309-40d5-8922-004f831f5d38",
        # )
        #     text = state['transcript']
        #     text_length = len(state['transcript'])

        # # Split the text into chunks of 4096 characters
        #     chunk_size = 4096
        #     chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        llm_chain = promt | llm
        con = llm_chain.invoke({"text": state["summarized_text"]}).content
        state["translated_str"] = con
        print(state["translated_str"])
        return state
    else:
        pass

    return state
import secrets
import string


def recomadntion_eval(state: AgentState) -> AgentState:
    print("in rec")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index_name = "final"
    pc = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    list_of_titles = []
    list_of_content = []
    for i in range(5):

        doc = pc.similarity_search(
            query=state["summarized_text"],
            filter={"title": {"$nin": list_of_titles}, "url": {"$ne": state["input"]}},
            k=1,
        )
        list_of_titles.append(doc[0].metadata["title"])
        list_of_content.append(doc[0])

    system = """youre an agent that will compare the input_transcript with a given text and give it a realtion score from 0 to 10 . just give me the score and dont explain anything . youre answer should be the score only as follwing : (socre)   


    input_transcript: {input_transcript}       
    given text : {given_text}     



    youre answer :              
    """
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{given_text}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    answer_grader = answer_prompt | llm | StrOutputParser()
    scored_list = []
    for rec in list_of_content:
        print(rec.page_content)
        socre = answer_grader.invoke(
            {"input_transcript": state["transcript"], "given_text": rec.page_content}
        )
        scored_list.append((rec.metadata["title"], rec.metadata["url"], socre))
    print(scored_list)

    max_tuple = max(scored_list, key=lambda x: x[2])
    return {"rec": max_tuple}


def text_to_speech_stream(state: AgentState) -> AgentState:
    ELEVENLABS_API_KEY = ""
    client = ElevenLabs(
        api_key=ELEVENLABS_API_KEY,
    )
    tit=re.sub(r'[^a-zA-Z]', '', state['title'])
    if state["translate_op"] == "1":
        audio = client.generate(
            text=state["translated_str"],
            voice="Rachel",
            model="eleven_multilingual_v2",
            stream=False,
        )   
        
        print(tit)
        print(tit)
        print(tit)
        print(tit)
        print(tit)
        save(audio, os.path.join(settings.MEDIA_ROOT, f"{tit}.mp3"))
    else:
        audio = client.generate(
            text=state["summarized_text"],
            voice="Rachel",
            model="eleven_multilingual_v2",
            stream=False,
        )
        save(audio, os.path.join(settings.MEDIA_ROOT,f"{tit}.mp3"))

    return state


def pp(state):  
    print(state["summarized_text"])
    print(state["translate_op"])
    print(state["translated_str"])

    return state


def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("loader", loader)
    workflow.add_node("summary", summary)
    workflow.add_node("translate", translate)
    workflow.add_node("recomadntion_eval", recomadntion_eval)
    workflow.add_node("text_to_speech_stream", text_to_speech_stream)

    # workflow.add_edge("loader", "safety_check")
    workflow.add_conditional_edges(
        "loader", safety_check, {"0": END, "1": "summary"}
    )
    workflow.add_edge("summary", "translate")
    workflow.add_edge("translate", "recomadntion_eval")
    workflow.add_edge("recomadntion_eval", "text_to_speech_stream")



    workflow.set_entry_point("loader")

    return workflow.compile()


def mainx(inputurl, tr):
    graph = create_graph()
    a = graph.invoke({"input": inputurl, "translate_op": tr})
    return a


import time


def openai_example(request: HttpRequest):
    for key in list(request.session.keys()):
        request.session[key] = None
    res = None
    audio_url = None
    if request.method == "POST":
        characters = string.ascii_letters + string.digits
        random_string = ''.join(secrets.choice(characters) for i in range(7))
        mainx(inputurl=request.POST["prompt"], tr=0)

        with open(os.path.join(settings.MEDIA_ROOT, f"{random_string}.mp3"), "rb") as f:
            audio_content = f.read()
        audio_url = os.path.join(settings.MEDIA_URL, f"{random_string}.mp3")

        print(audio_url)
        # Prepare response

    return render(request, "Tuq.html", {"response": res, "audio_file": audio_url})


def link_upload(request: HttpRequest):
    if request.method == "POST":
        
        print(request.POST.get("urlInput"))
        url = request.POST.get("urlInput")
        a = None
        if url:
            print(request.POST.get("option"))
            res = None

            a = mainx(inputurl=request.POST["urlInput"], tr=request.POST["option"])
            print(a)
            if not a.get('safety'):
                return  render(request , 'unsfae.html')
            tit=re.sub(r'[^a-zA-Z]', '', a['title'])
            print(tit)
            print(tit)
            print(tit)
            print(tit)
            print(tit)
            with open(os.path.join(settings.MEDIA_ROOT, f"{tit}.mp3"), "rb") as f:
                audio_content = f.read()
            audio_url = os.path.join(settings.MEDIA_URL, f"{tit}.mp3")
            print(audio_url)
            summary_text = (
                a["translated_str"] if a["translated_str"] else a["summarized_text"]
            )
            
            title = a["title"]
            rec = a["rec"]
            thumbnail_url = a["thumbnail_url"]
            rec_tumb = YoutubeLoader.from_youtube_url(
            rec[1], add_video_info=True, language=["en", "ar"]
    )       
            rec_tumb=rec_tumb.load()
            rec_tumb=rec_tumb[0].metadata['thumbnail_url']

        context = {
            "transcript": a["transcript"],
            "response": res,
            "audio_file": audio_url,
            "summary": summary_text,
            "title": title,
            "rec": rec,
            "thumbnail_url": thumbnail_url,
            'url':url,
            'rec_tumb':rec_tumb
        }
        for key, value in context.items():
            request.session[key] = value
        return render(
            request,
            "Tuq.html",
            {
                "response": res,
                "audio_file": audio_url,
                "summary": summary_text,
                "title": title,
                "rec": rec,
                "thumbnail_url": thumbnail_url,
                'url':url,
                'rec_tumb':rec_tumb
            },
        )

    return render(request, "firstpage.html")


def chat(request: HttpRequest):
    print("oustide")
    if request.method == "POST":
        print(request.session["title"])
        print("inside")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = PromptTemplate.from_template(
            """
                                            answer the question only with the given context: 
                                            context : {context} \n the end of context\n
                                            question:{question} \n the end of question \n 
                                            your answer : 
                                            """
        )
        chain = prompt | llm

        answer = chain.invoke(
            {
                "context": request.session["transcript"],
                "question": request.POST["question"],
            }
        ).content
        print(answer)
        return render(
            request,
            "Tuq.html",
            {
                "response": request.session["response"],
                "audio_file": request.session["audio_file"],
                "summary": request.session["summary"],
                "title": request.session["title"],
                "rec": request.session["rec"],
                "answer": answer,
                "thumbnail_url": request.session["thumbnail_url"],
                "question": request.POST["question"],
                'url':request.session['url'],
                'rec_tumb':request.session['rec_tumb'],

            },
        )
    print(request.session["title"])
    return render(
        request,
        "Tuq.html",
        {
            "response": request.session["response"],
            "audio_file": request.session["audio_file"],
            "summary": request.session["summary"],
            "title": request.session["title"],
            "rec": request.session["rec"],
            "answer": "answer",
            "thumbnail_url": request.session["thumbnail_url"],
            "question": request.POST["question"],
            'url':request.session['url'],
            'rec_tumb':request.session['rec_tumb'],

        },
    )
