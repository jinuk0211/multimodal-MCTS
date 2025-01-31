import random
from tasks.science import SearchTask
from MCTS.base import treeNode
from models.get_response import *
from MCTS.mcts import MCTS
from utils.verify_MATH import exact_match_score, grade_answer, extract_answer
from utils.verify_llm import llm_verify
from utils.solution_summary_extractor import extract_summary_from_solution


class MCTS_Task(SearchTask):
    def __init__(self, data, propose_method='glm', value_method='glm', branch=3, end_gate=0.9, roll_policy='greedy',
                 roll_branch=1, roll_forward_steps=3, time_limit=None, iteration_limit=None, exploration_constant=0.7,
                 alpha=0.5, inf=1.0, temperature=0.7, max_tokens=2048, seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=256, use_case_prompt=False, use_reflection='simple', low=0, high=1,
                 evaluate='', sample_value='simple', answer=None, verify_method='string', lang='zh', weighted_verify=False):
        super().__init__(data, propose_method, value_method)
# class SearchTask(object):
#     def __init__(self, data, propose_method='glm', value_method='glm'):
#         super().__init__()
#         self.question = data
#         self.propose_method = propose_method
#         self.value_method = value_method
#         self.value_cache = {}                
        assert 0 <= low < high, "Inappropriate value range!"
        self.mode = 'mcts'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.branch = branch
        self.use_case_prompt = use_case_prompt
        self.low = low
        self.high = high
        self.evaluate = evaluate
        self.end_gate = end_gate
        self.use_reflection = use_reflection
        self.roll_policy = roll_policy
        self.roll_branch = roll_branch
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant
        self.roll_forward_steps = roll_forward_steps
        self.alpha = alpha
        self.limit_type = None
        self.INF = inf
        self.node_count = 1
        self.sample_value = sample_value
        self.answer = answer
        self.verify_method = verify_method
        self.reward_model_type = 'prm' if USE_PRM else 'vm'
        self.lang = lang
        self.weighted_verify = weighted_verify

    def update_count(self):
        self.node_count += 1

    def clear_cache(self):
        self.value_cache = {}
        self.node_count = 1

    def set_limit_type(self):
        if self.time_limit is not None:
            if self.iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.limit_type = 'time'
        else:
            if self.iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if self.iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.limit_type = 'iterations'

    def get_next_step(self, y, step_n):
        if self.use_case_prompt:
            prompt = self.single_propose_prompt_wrap(self.question, y, step_n)
    # @staticmethod
    # def single_propose_prompt_wrap(x: str, y: str = '', step: int = 0) -> str:
    #     print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
    #     print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '위의 단계를 기반으로, 현재 단계에 가능한 솔루션은 다음과 같습니다:\n')
    #     prompt = single_proposal_prompt + x + '\n已有步骤:\n' + y + '\n输出:'
    #     return prompt          

# 이떄의 single_proposal_prompt는 
# 당신의 임무는 주어진 이과 문제와 기존의 해결 단계(완전한 정답이 아님)를 바탕으로 올바른 다음 단계를 제시하는 것입니다. 아래는 몇 가지 예제입니다. 학습하세요.
#few shot 몇개의 예시
#이와 같은 방식으로 주어진 문제의 기존 해결 단계를 기반으로 올바른 다음 단계를 도출하는 것이 목표입니다.
# proposal_prompt + (x는 question) + 이전단계:\n + (y가 이전단계) + "결론": 

        else:
            if self.propose_method == 'gpt':
                prompt = self.zero_single_propose_wrap_gpt(self.question, y, step_n, self.lang)
            elif self.propose_method == 'mistral' or self.propose_method == 'llama':
                prompt = self.zero_single_propose_wrap_mistral(self.question, y, step_n)
            else:
                prompt = self.zero_single_propose_wrap(self.question, y, step_n, self.lang)

        response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, self.max_new_tokens)
# def get_proposal(prompt, method='glm', temperature=0.7, max_tokens=2048, seed=170, max_length=2048, truncation=True,
#                  do_sample=True, max_new_tokens=1024):
#     response = []
#     cnt = 2
#     if method == 'glm':
#         while not response and cnt:
#             response = glm(prompt, BASE_MODEL_GLM, temperature=temperature, max_tokens=max_tokens, seed=seed)
#         return response
        if not response:
            print('다음 단계를 가져오지 못했습니다！\n')
            return ''

        if len(response) > 5:
            response = response[:5]

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()
#strip함수 "  Hello, World!  " -> "Hello, World!"
        if self.lang == 'zh':
            if '下一步:' in p: #다음단계
                stp = p.split('下一步:')[1].strip()  #s_{i+1} 도출
                if len(stp) < 2:
                    print('输出步骤过短！\n') #step 수가 너무 적은데 답을 도출해냄
                    return ''
                if stp in y: # get_next_step(self, y, step_n):의 솔루션 str
                    print('输出步骤重复！\n') #중복이 있음
                    return ''

                revised_ = '步骤' + str(step_n) + ':' + stp
                print(f'标准化后新的步骤:{revised_}\n') # 표준화(일관된 형식으로 정리)한 후, 그에 따라 새롭게 정리된 단계(스텝)
                return revised_ + '\n'

            elif '步骤' in p and ':' in p: #步骤 == 다음단계
                pre_len = len(p.split(':')[0])
                p_ = p[pre_len:]
                p_ = p_.split('步骤')[0].strip()
                if len(p_) < 3:
                    print('step 수가 너무 적은데 답을 도출해냄！\n')
                    return ''
                if p_[1:] in y:
                    print('输出步骤重复！\n')
                    return ''

                revised_ = '步骤' + str(step_n) + p_
                print(f'标准化后新的步骤:{revised_}\n')
                return revised_ + '\n'

            else:
                print('输出格式有误！\n')
                return ''

        else:
            if "Next step:" in p:
                stp = p.split('Next step:')[1].strip()
                if len(stp) < 2:
                    print('输出步骤过短！\n')
                    return ''
                if stp in y:
                    print('输出步骤重复！\n')
                    return ''

                revised_ = 'Step ' + str(step_n) + ': ' + stp
                print(f'标准化后新的步骤:{revised_}\n')
                return revised_ + '\n'

            elif "Step" in p and ":" in p:
                pre_len = len(p.split(':')[0])
                p_ = p[pre_len:]
                p_ = p_.split('Step')[0].strip()
                if len(p_) < 4:
                    print('输出步骤过短！\n')
                    return ''
                p_ = p_[1:].strip()
                if p_ in y:
                    print('输出步骤重复！\n')
                    return ''

                revised_ = 'Step ' + str(step_n) + ': ' + p_
                print(f'标准化后新的步骤:{revised_}\n')
                return revised_ + '\n'

            else:
                p_ = p.strip()
                if len(p_) < 3:
                    print('输出步骤过短！\n')
                    return ''
                if p_ in y:
                    print('输出步骤重复！\n')
                    return ''

                revised_ = 'Step ' + str(step_n) + ': ' + p_
                print(f'标准化后新的步骤:{revised_}\n')
                return revised_ + '\n'

# reflection 프롬프트
# Your task is to give the correct next step, given a science problem, an existing partial solution (not a complete answer) and some analysis for the next step.
# Assuming the input is n-steps, then the format of the input is:
# "Problem: ...
# Existing Steps:
# Step 1: ...
# Step 2: ...
# ...
# Step n: ...
# Analysis: ..."

# where ... denotes omitted input information.
# If no existing steps are provided, you need to output the first step referring to the given analysis. Otherwise, you need to output the next step (step n+1) that you think is correct, following the ideas of the existing steps and provided analysis.
# The output format is limited to:
# "Next step: ..."
# where ... indicates omitted output information, which is the part you should fill in. Your output should be a complete reasoning step that includes calculations, reasoning, choosing answers, etc.
# Here is the input, please follow the restricted output format.

# Problem: '''
    def get_next_step_use_reflection(self, y, step_n, reflection):  # 暂不支持 case-prompt
        if self.propose_method == 'gpt' or self.propose_method == 'local':
            propose_prompt = self.zero_single_propose_wrap_use_reflection_gpt(self.question, y, step_n, reflection,
                                                                              self.lang)
        else:
            propose_prompt = self.zero_single_propose_wrap_use_reflection(self.question, y, step_n, reflection,
                                                                          self.lang)
        response = get_proposal(propose_prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, self.max_new_tokens)
        if not response:
            print('다음 단계를 가져오지 못했습니다！\n')
            return ''

        if len(response) > 5:
            response = response[:5]

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        if self.lang == 'zh':
            if '下一步:' in p:
                stp = p.split('下一步:')[1].strip()
                if len(stp) < 2:
                    print('输出步骤过短！\n')
                    return ''
                if stp in y:
                    print('输出步骤重复！\n')
                    return ''

                revised_ = '步骤' + str(step_n) + ':' + stp
                print(f'标准化后新的步骤:{revised_}\n')
                return revised_ + '\n'

            elif '步骤' in p and ':' in p:
                pre_len = len(p.split(':')[0])
                p_ = p[pre_len:]
                p_ = p_.split('步骤')[0].strip()
                if len(p_) < 3:
                    print('输出步骤过短！\n')
                    return ''
                if p_[1:] in y:
                    print('输出步骤重复！\n')
                    return ''

                revised_ = '步骤' + str(step_n) + p_
                print(f'标准化后新的步骤:{revised_}\n')
                return revised_ + '\n'

            else:
                print('输出格式有误！\n')
                return ''

        else:
            if "Next step:" in p:
                stp = p.split('Next step:')[1].strip()
                if len(stp) < 2:
                    print('输出步骤过短！\n')
                    return ''
                if stp in y:
                    print('输出步骤重复！\n')
                    return ''

                revised_ = 'Step ' + str(step_n) + ': ' + stp
                print(f'标准化后新的步骤:{revised_}\n')
                return revised_ + '\n'

            elif "Step" in p and ":" in p:
                pre_len = len(p.split(':')[0])
                p_ = p[pre_len:]
                p_ = p_.split('Step')[0].strip()
                if len(p_) < 4:
                    print('输出步骤过短！\n')
                    return ''
                p_ = p_[1:].strip()
                if p_ in y:
                    print('输出步骤重复！\n')
                    return ''

                revised_ = 'Step ' + str(step_n) + ': ' + p_
                print(f'标准化后新的步骤:{revised_}\n')
                return revised_ + '\n'

            else:
                print('输出格式有误！\n')
                return ''

    def get_simple_reflection(self, y, step_n):
        if step_n == 1:
            return '<continue>'
        if self.propose_method in ['local', 'mistral', 'llama'] and self.lang == 'en':
            if 'answer is' in y or '\\boxed' in y:
                return '<end>'

        if self.propose_method == 'mistral':
            reflection_prompt = self.single_reflection_wrap_simple_mistral(self.question, y, step_n)
# single_reflection_prompt_simple_mistral = '''
# Given a science problem and some corresponding steps, if the given steps have already solved the problem and provided the final answer to the question, then you should output: "solved". Otherwise, please output: "unsolved".
# Following the instruction, output "unsolved" or "solved", with no other information.
# Problem: '''        
        else:
            reflection_prompt = self.single_reflection_wrap_simple(self.question, y, step_n, self.lang)
        cnt = 3
        response = []
        while not response and cnt:
            response = get_proposal(reflection_prompt, self.propose_method, self.temperature, self.max_tokens,
                                    self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, 128)
            cnt -= 1
        if not response:
            print('获得意见失败！\n')
            return '<end>'

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        if self.lang == 'zh':
            if '已解决' in p or '已经解决' in p:
                if step_n > 1:
                    print('此步问题已解决，停止下探。\n')
                    print('标准化后的意见: <end>\n')
                    return '<end>'
            print('标准化后的意见: <continue>\n')
            return '<continue>'

        if self.lang == 'ko':  
            if '해결됨' in p or '이미 해결됨' in p:  
                if step_n > 1:  
                    print('이 단계의 문제가 해결되었으므로 더 이상 진행하지 않습니다.\n')  
                    print('표준화된 의견: <end>\n')  
                    return '<end>'  
            print('표준화된 의견: <continue>\n')  
            return '<continue>'  
    

        else:
            if 'unsolved' in p or step_n <= 1:
                print('표준화된 의견: <continue>\n') 
                return '<continue>'
            elif 'solved' in p:
                print('标准化后的意见: <end>\n')
                return '<end>'
            else:
                print('标准化后的意见: <continue>\n')
                return '<continue>'

    def get_reflection(self, y, step_n):
        if self.propose_method in ['local', 'mistral', 'llama'] and self.lang == 'en':
            if 'answer is' in y or '\\boxed' in y:
                return '<end>'

        if self.lang == 'zh':
            if self.propose_method == 'gpt' or self.propose_method == 'local':
                reflection_prompt = self.single_reflection_wrap_gpt(self.question, y, step_n)
            elif self.propose_method == 'llama':
                reflection_prompt = self.single_reflection_wrap_llama(self.question, y, step_n)
            else:
                reflection_prompt = self.single_reflection_wrap(self.question, y, step_n, self.lang)
        else:
            reflection_prompt = self.single_reflection_wrap(self.question, y, step_n, self.lang)

        cnt = 3
        response = []
        while not response and cnt:
            response = get_proposal(reflection_prompt, self.propose_method, self.temperature, self.max_tokens,
                                    self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, self.max_new_tokens)
            cnt -= 1
        if not response:
            print('获得意见失败！\n')
            return ''

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        if self.lang == 'zh':
            if '已解决' in p or '已经解决' in p:
                if step_n > 1:
                    print('此步问题已解决，停止下探。\n')
                    return '<end>'
                else:
                    return ''

            if '意见:' not in p:
                print('输出格式有误！\n')
                return ''
            revised_ = p.split('意见:')[1]
            print(f'标准化后的意见:{revised_}\n')
            return revised_
            
        if self.lang == 'ko':  
            if '해결됨' in p or '이미 해결됨' in p:  
                if step_n > 1:  
                    print('이 단계의 문제가 해결되었으므로 더 이상 진행하지 않습니다.\n')  
                    return '<end>'  
                else:  
                    return ''  

            if '의견:' not in p:  
                print('출력 형식이 올바르지 않습니다!\n')  
                return ''  
            revised_ = p.split('의견:')[1]  
            print(f'표준화된 의견: {revised_}\n')  
            return revised_  

        else:
            if 'Problem solved' in p:
                print('标准化后的意见: <end>\n')
                return '<end>'
            else:
                if 'Analysis:' not in p:
                    print('输出格式有误！\n')
                    return ''
                revised_ = p.split('Analysis:')[1].strip()
                print(f'标准化后的意见:{revised_}\n')
                return revised_

    def get_step_value(self, y):
        if y in self.value_cache.keys():
            return self.value_cache[y]

        if self.value_method == 'local':
            if self.lang == 'zh':
                prompt_answer = '问题:' + self.question + '\n步骤:\n' + '【答案】' + y
            if self.lang == 'ko':
                prompt_answer = '문제:' + self.question + '\n답안과정:\n' + y
            else:
                prompt_answer = 'Problem: ' + self.question + '\nSolution:\n' + y
            value = get_value(prompt_answer, self.value_method, self.temperature, self.max_tokens, self.seed,
                              self.max_length, self.low, self.high)
    # method == 'local':
    #     value = low
    #     while cnt:
    #         try:
    #             value = local_value_model(prompt_answer, max_length=max_length, low=low, high=high)
    #             break
    #         except Exception as e:
    #             print(f'obtain<{method}>score fail!\nError:{e}\n')
    #             cnt -= 1
    #     return value
            
            print(f'获得评分:{value}\n') #평가 받기
            self.value_cache.update({y: value})
            return value

        else:
            prompt = self.value_prompt_wrap(self.question, y)
            response = get_value(prompt, self.value_method, self.temperature, self.max_tokens, self.seed,
                                 self.max_length, self.low, self.high)
            value = self.value_outputs_unwrap(response, self.low, self.high)
            print(f'获得评分:{value}\n') #평가받기
            self.value_cache.update({y: value})
            return value

    def get_summary(self, y):
        if self.lang == 'zh':
            if self.evaluate == 'scibench':
                prompt = self.evaluate_summary_prompt_wrap(self.question, y)
            elif self.evaluate == 'scieval':
                prompt = self.general_evaluate_summary_prompt_wrap(self.question, y)
            else:
                prompt = self.summary_prompt_wrap(self.question, y)

            response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, 128)

            if not response:
                print('获得综述失败！\n') # 요약을 받지 못했습니다
                return ''
            p = ''
            for _ in response:
                p = p + _ + ' '
            p = p.strip()

            if self.evaluate:
                if len(p) < 1:
                    print('获得综述过短！\n') #요약된 내용이 너무 짧습니다
                    return ''

                if '综上所述，最终答案是:' not in p: #결론적으로, 최종 답은
                    summ = '综上所述，最终答案是:' + p
                    print(f'获得综述:{summ}\n') #요약된 내용
                    return summ
                else:
                    summ = '综上所述，最终答案是:' + p.split('综上所述，最终答案是:')[-1]
                    print(f'获得综述:{summ}\n')
                    return summ

            else:
                if len(p) < 1:
                    print('获得综述过短！\n')
                    return ''

                p = p.replace('综上所述,', '综上所述，')
                if '综上所述，' not in p:
                    summ = '综上所述，' + p
                    print(f'获得综述:{summ}\n')
                    return summ
                else:
                    summ = '综上所述，' + p.split('综上所述，')[-1]
                    print(f'获得综述:{summ}\n')
                    return summ

        else:
            prompt = self.MATH_summary_prompt_wrap(self.question, y)
            response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, 128)
            if not response:
                print('获得综述失败！\n')
                return ''
            p = ''
            for _ in response:
                p = p + _
            summ = p.strip()
            print(f'获得综述:{summ}\n') #요약 받기

            return summ

    def get_MATH_summary(self, y):
        prompt = self.MATH_summary_prompt_wrap(self.question, y)
        response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, 128)
        if not response:
            print('获得综述失败！\n')
            return ''
        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        print(f'获得综述:{p}\n')
        return p

    def verify_end_nodes(self, root):
        if self.reward_model_type == 'vm':
            end_leaf_nodes = root.get_all_end_root_nodes_vm(self.end_gate)
        else:
            end_leaf_nodes = root.get_all_end_root_nodes_prm()
        flag = False
        for leaf in end_leaf_nodes:
            leaf.on_final_route = True
            cnt = 5
            summ = ''
            while cnt:
                if self.verify_method == 'string':
                    summ = self.get_MATH_summary(leaf.y)
                else:
                    summ = self.get_summary(leaf.y)
                if summ:
                    leaf.summary = summ
                    break
                else:
                    cnt -= 1
            if not summ:
                summ = extract_summary_from_solution(leaf.y)
                leaf.summary = summ

            if self.verify_method == 'string':
                result = exact_match_score(summ, self.answer)
            else:
                result = llm_verify(summ, self.answer)
            if result:
                if self.reward_model_type == 'vm':
                    leaf.min_steps_to_correct = 1
                else:
                    leaf.he = 1
                flag = True
        return flag, end_leaf_nodes

    def get_final_solution(self, root, weighted):  # for evaluation
        if self.reward_model_type == 'vm':
            end_leaf_nodes = root.get_all_end_root_nodes_vm(self.end_gate)
        else:
            end_leaf_nodes = root.get_all_end_root_nodes_prm()

        if not end_leaf_nodes or not weighted:
            if not end_leaf_nodes:
                best_node, best_V = root.getBestV()
            else:
                sorted_nodes = sorted(end_leaf_nodes, key=lambda x: x.V, reverse=True)
                best_node = sorted_nodes[0]
            solution = best_node.y
            cnt = 5
            summ = ''
            while cnt:
                if self.verify_method == 'string':
                    summ = self.get_MATH_summary(solution)
                else:
                    summ = self.get_summary(solution)
                if summ:
                    best_node.summary = summ
                    break
                else:
                    cnt -= 1
            if not summ:
                summ = extract_summary_from_solution(solution)
                best_node.summary = summ
            return solution, summ

        else:
            all_answers = {}  # {answer: [solution, summ, value]}
            for leaf in end_leaf_nodes:
                cnt = 5
                summ = ''
                while cnt:
                    if self.verify_method == 'string':
                        summ = self.get_MATH_summary(leaf.y)
                    else:
                        summ = self.get_summary(leaf.y)
                    if summ:
                        leaf.summary = summ
                        break
                    else:
                        cnt -= 1
                if not summ:
                    summ = extract_summary_from_solution(leaf.y)
                    leaf.summary = summ

                extracted_answer = extract_answer(summ)
                if extracted_answer in all_answers.keys():
                    all_answers[extracted_answer][2] += leaf.V
                else:
                    all_answers[extracted_answer] = [leaf.y, summ, leaf.V]

            best_answer = max(all_answers.values(), key=lambda x: x[2])
            solution = best_answer[0]
            summ = best_answer[1]
            return solution, summ

    def run(self):
        self.clear_cache()
        self.set_limit_type()
        node, finish, root = MCTS(self)
        # vm
        if self.reward_model_type == 'vm':
            if self.sample_value != 'full':
                if self.evaluate == 'scibench':  # SciBench style
                    solution = node.y
                    summary = self.get_summary(solution)
                    final_answer = {'content': self.question, 'solution': solution, 'summary': summary,
                                    'finish': finish}
                    if self.sample_value == 'simple':
                        node.trace_route()
                        new_value_samples = node.get_new_value_samples()
                        final_answer.update({'value_samples': new_value_samples})
                else:  # MATH style
                    solution = node.y
                    cnt = 5
                    summ = ''
                    while cnt:
                        if self.verify_method == 'string':
                            summ = self.get_MATH_summary(solution)
                        else:
                            summ = self.get_summary(solution)
                        if summ:
                            node.summary = summ
                            break
                        else:
                            cnt -= 1

                    if not summ:
                        summ = extract_summary_from_solution(solution)
                        node.summary = summ

                    result = exact_match_score(summ, self.answer)
                    final_answer = {'content': self.question, 'solution': solution, 'summary': summ, 'finish': finish,
                                    'accurate': result, 'real_answer': self.answer}
                return final_answer, root
            else:
                if not self.evaluate:  # generate only
                    assert self.answer is not None, 'Answer is None!\n'
                    flag, end_leaf_nodes = self.verify_end_nodes(root)

                    # extract policy data
                    new_policy_samples = []
                    for leaf in end_leaf_nodes:
                        solution = leaf.y
                        summ = leaf.summary
                        correct = True if leaf.min_steps_to_correct == 1 else False
                        new_policy_sample = {'solution': solution, 'summary': summ, 'correct': correct}
                        new_policy_samples.append(new_policy_sample)

                    # extract value data
                    if flag:
                        new_value_samples = root.get_full_value_samples_vm(end_leaf_nodes)
                    else:
                        new_value_samples = []
                    final_answer = {'content': self.question, 'policy_samples': new_policy_samples,
                                    'value_samples': new_value_samples, 'real_answer': self.answer}
                    return final_answer, root
                else:
                    assert self.answer is not None, 'Answer is None!\n'
                    solution, summ = self.get_final_solution(root, self.weighted_verify)
                    if not summ:
                        result = False
                    else:
                        result = exact_match_score(summ, self.answer)
                    final_answer = {'content': self.question, 'solution': solution, 'summary': summ, 'finish': finish,
                                    'accurate': result, 'real_answer': self.answer}
                    return final_answer, root

        # prm (only sample generation available now)
        else:
            assert self.sample_value, 'Only sampling is supported for prm!\n'
            assert self.answer is not None, 'Answer is None!\n'
            flag, end_leaf_nodes = self.verify_end_nodes(root)

            # extract policy data
            new_policy_samples = []
            for leaf in end_leaf_nodes:
                solution = leaf.y
                summ = leaf.summary
                correct = True if leaf.he == 1 else False
                new_policy_sample = {'solution': solution, 'summary': summ, 'correct': correct}
                new_policy_samples.append(new_policy_sample)

            # extract value data
            if flag:
                new_value_samples = root.get_full_value_samples_prm(end_leaf_nodes)
            else:
                new_value_samples = []
            final_answer = {'content': self.question, 'policy_samples': new_policy_samples,
                            'value_samples': new_value_samples, 'real_answer': self.answer}
            return final_answer, root
