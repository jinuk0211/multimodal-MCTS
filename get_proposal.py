from model import *


# given prompt, generate proposal under instruction, unwrap is required
def get_proposal(prompt, method='glm', temperature=0.7, max_tokens=2048, seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=1024):
    response = []
    cnt = 2
    if method == 'glm':
        while not response and cnt:
            response = glm(prompt, BASE_MODEL_GLM, temperature=temperature, max_tokens=max_tokens, seed=seed)

# def glm(prompt, model=BASE_MODEL_GLM, temperature=0.7, max_tokens=1000, seed=170) -> list:
#     return get_glm_reply(prompt, model, temperature=temperature, max_tokens=max_tokens, seed=seed)
          
            cnt -= 1
        if not response:
            print(f'obtain<{method}>response fail!\n')
            return []
        return response

    elif method == 'gpt':
        while not response and cnt:
            response = gpt(prompt, model=BASE_MODEL_GPT, temperature=temperature, max_tokens=max_tokens)
            cnt -= 1
        if not response:
            print(f'obtain<{method}>response fail!\n')
            return []
        return response

# def gpt(prompt, model=BASE_MODEL_GPT, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
#     messages = [{"role": "user", "content": prompt}]
#     out = []
#     cnt = 5
#     while cnt:
#         try:
#             out = chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)[
#                 0].split('\n')
#             break
#         except Exception as e:
#             print(f"Error occurred when getting gpt reply!\nError type:{e}\n")
#             cnt -= 1
                   
#     return out


# def chatgpt(messages, model=BASE_MODEL_GPT, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
#     global completion_tokens, prompt_tokens
#     outputs = []
#     while n > 0:
#         cnt = min(n, 20)
#         n -= cnt
#         res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
#                                        n=cnt, stop=stop)
#         # print(f'得到GPT回复:{res}\n\n')
#         outputs.extend([choice["message"]["content"] for choice in res["choices"]])
#         # log completion tokens
#         completion_tokens += res["usage"]["completion_tokens"]
#         prompt_tokens += res["usage"]["prompt_tokens"]
#     return outputs
    elif method == 'llama' or method == 'mistral' or method == 'local':
        while not response and cnt:
            response = local_inference_model(prompt, max_length=max_length, truncation=truncation, do_sample=do_sample,
                                             max_new_tokens=max_new_tokens, temperature=temperature)
            cnt -= 1
        if not response:
            print(f'obtain<{method}>response fail!\n')
            return []
        return response

    else:
        print('This method of getting responses is not yet supported!\n')
        return []
      
# def local_inference_model(query, max_length=2048, truncation=True, do_sample=False, max_new_tokens=1024,
#                           temperature=0.7):
#     assert INFERENCE_LOCAL, "Inference model not implemented!\n"
#     if inference_type == 'glm':
#         return get_local_response(query, inference_model, inference_tokenizer, max_length=max_length,
#                                   truncation=truncation,
#                                   do_sample=do_sample, max_new_tokens=max_new_tokens, temperature=temperature)

# given prompt + answer, find its value
# if you use api, unwrap is required. if you use local value model, the value is directly obtained
def get_value(prompt_answer, method='glm', temperature=0.7, max_tokens=1000, seed=170, max_length=2048, low=0, high=1):
    response = []
    cnt = 2
    if method == 'glm':
        while not response and cnt:
            response = glm(prompt_answer, BASE_MODEL_GLM, temperature=temperature, max_tokens=max_tokens, seed=seed)
            cnt -= 1
        if not response:
            print(f'obtain<{method}>score fail!\n')
            return []
        return response

    elif method == 'gpt':
        while not response and cnt:
            response = gpt(prompt_answer, model=BASE_MODEL_GPT, temperature=temperature, max_tokens=max_tokens)
            cnt -= 1
        if not response:
            print(f'obtain<{method}>score fail!\n')
            return []
        return response

    elif method == 'local':
        value = low
        while cnt:
            try:
                value = local_value_model(prompt_answer, max_length=max_length, low=low, high=high)
                break
            except Exception as e:
                print(f'obtain<{method}>score fail!\nError:{e}\n')
                cnt -= 1
        return value

    else:
        print('This method of getting scores is not yet supported!\n')
        return []
