import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = ''

# query = input('입력하세요')


# prompt = """
# 다음 문장이 긍정이면 positive, 부정이면 negative를 만들어라

# text: 이 영화 최악이다
# sentiment: negative

# text: 배우들이 연기를 너무 잘하네
# sentiment: positive

# text: """

# prompt = prompt + query + '\nsentiment: '


# chat_completion = openai.Completion.create(
#     model="text-davinci-003", prompt=prompt)

# print(chat_completion['choices'][0]['text'])
system_instruction = """
너는 햄버거 가게 AI 비서야

아래는 햄버거 종류야, 아래 종류의 버거 말고는 다른 버거는 없어

- 빅맥
- 쿼터파운드
- 치즈버거

위의 메뉴 말고는 없다고 생각하면돼
"""


messages = [{'role': 'system', 'content': system_instruction}]


def ask(text):
    user_input = {'role': 'user', 'content': text}
    messages.append(user_input)

    reponse = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)

    resp = reponse.to_dict_recursive()
    bot_text = reponse['choices'][0]['message']['content']
    bot_resp = {'role': 'assistant', 'content': bot_text}

    messages.append(bot_resp)

    return bot_text


while True:
    user_input = input('user input: ')
    bot_resp = ask(user_input)

    print('-' * 30)
    print(f'user_input: {user_input}')
    print(f'bot_resp: {bot_resp}')
