import openai
from transformers import pipeline
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import gradio as gr

# Specify the key
with open("key.txt") as f:
    openai.api_key = f.read().split("\n")[0]

class MathAssistant:
    
    def __init__(self):
        
        self.template = """Your name is Matt, a virtual assistant for answering simple grade school math word problems involving simple operators like addition , subtraction, division and multipication.
        If the math problem asked is more complex than what specified above, like if it involves concepts from probability or calculus, try answering that but warn the user that it might be out of your capabilities and you advise the user to take help from their teachers or parents.
        
        This is the only thing that you are capable of. If the user asks about some topic other than math, just say that you are only Math Assistant and it is out of your capability to answer the question.
        
        A few examples of a potential conversation can be:
        
        Human: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
        
        Matt: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
        
        Human: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"
        
        Matt: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29..
        
        Human: What if there were 25 computers in the server room in the beginning.
        
        Matt: If there were 25 computers in the beginning, the total will be 20 + 25 = 45. The answer is 45.
        
        {history}
        
        Human: {input}
        Matt:"""
        
        self.history = ""

    def reset(self):
        self.history = ""
        
    def __call__(self, question):
        prompt = self.template.replace("{history}", self.history)
        prompt = prompt.replace("{input}", question)
        print(prompt)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=100
        )
        response =  response["choices"][0]["text"].strip()
        
        # Update history
        self.history += f"\n\nHuman: {question}\n\nMatt: {response}"
        return response

class MathAssistantWithLangChain(MathAssistant):
    
    def __init__(self, max_memory = 2, verbose = False):
        super().__init__()
        
        self.verbose = verbose
        self.max_memory = max_memory
        self.template = PromptTemplate(
            input_variables = ["history", "input"],
            template=self.template
        )
    
        self.llm_chain = LLMChain(
            llm=OpenAI(
                model_name="text-davinci-003",
                temperature=0,
                max_tokens=100,
                openai_api_key = openai.api_key
            ),
            prompt=self.template,
            verbose=self.verbose,
            memory=ConversationBufferWindowMemory(k=max_memory)
        )
    
    def reset(self):
        self.llm_chain = LLMChain(
            llm=OpenAI(
                model_name="text-davinci-003",
                temperature=0,
                max_tokens=100,
                openai_api_key = openai.api_key
            ),
            prompt=self.template,
            verbose=self.verbose,
            memory=ConversationBufferWindowMemory(k=self.max_memory)
        )

    
    def __call__(self, question):
        return self.llm_chain.predict(input=question)

class MultilingualMathAssistant(MathAssistantWithLangChain):
    
    def __init__(self, lang, max_memory = 2, verbose = False):
        super().__init__(max_memory, verbose)
        
        self.lang = lang
        self.translation_pipeline = pipeline('text2text-generation', model="facebook/m2m100_418M")

    def translate_lang_to_en(self, text):
        
        return self.translation_pipeline(text, 
            forced_bos_token_id = self.translation_pipeline.tokenizer.get_lang_id('en'))[0]["generated_text"]
    
    def translate_en_to_lang(self, text):
        
        return self.translation_pipeline(text, 
            forced_bos_token_id = self.translation_pipeline.tokenizer.get_lang_id(self.lang))[0]["generated_text"]
    
    def __call__(self, question):
        
        # Translate question to English
        en_question = self.translate_lang_to_en(question)
        
        # Obtain the response in English
        en_response = super().__call__(en_question)
        
        # Translate the response to the bots language
        lang_response = self.translate_en_to_lang(en_response)
        
        return lang_response


def respond(chat_history, message):
    response = matt_hi(message)
    return chat_history + [[message, response]]        

# if __name__ == "__main__":
matt_hi = MultilingualMathAssistant("hi")
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(respond, [chatbot, msg], chatbot)
    clear.click(matt_hi.reset, None, chatbot, queue=False)

demo.launch()