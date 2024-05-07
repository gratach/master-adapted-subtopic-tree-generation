from openai import OpenAI
from random import randint

openaiClient = OpenAI()
def gpt_3_5_turbo_completion(query):
    answer = openaiClient.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": query
            }
        ],
        seed = randint(0, 1000000)
    )
    return answer.choices[0].message.content

def gpt_4_turbo_completion(query):
    answer = openaiClient.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": query
            }
        ],
        seed = randint(0, 1000000)
    )
    return answer.choices[0].message.content

def tryRecieveAnswer(query, completionFunction, answerConversion = lambda x: True, maxTries = 10):
    tryNumber = 0
    while tryNumber < maxTries:
        answer = completionFunction(query)
        try:
            answer = answerConversion(answer)
            return (answer, True)
        except:
            pass
        tryNumber += 1
    print(f"Failed to recieve answer for query: {query}")
    return (None, False)