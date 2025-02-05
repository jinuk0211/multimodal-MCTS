# multimodal-MCTS
{
  "content": "Calculate the sum of the first 10 prime numbers.",
  "answer": "129"
}
search_task -> MCTS_task -> MCTS_task.run -> MCTS -> MCTS_search
value prompt
```python
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        print('\n', '==============================', 'critic', '==============================', '\n')
        value_prompt = critic_simplified + x + '\n已有步骤:\n' + y.strip() + '\n输出:'
        return value_prompt

self_critic_prompt = '''
Given a science problem and an existing solution, your task is to evaluate the correctness of the solution and provide an evaluation score. 
Your output should be a decimal ranging from 0 to 1. The more correct the solution is, the higher your evaluation score should be.

Problem:'''
```
```python
from MCTS.task import *
question = "Calculate the sum of the first 10 prime numbers."
task = MCTS_Task(question, 'llama', 'local', lang='en')
output = task.run()
print(output['solution'])
```
```
python evaluate.py \
  --task_name "scibench" \
  --file "thermo" \
  --propose_method "gpt" \
  --value_method "local" \
  --mode "mcts" \
  --evaluate "scibench" \
  --iteration_limit 50 \
  --use_reflection "simple" \
  --branch 3
```
evaluate.py
```python
import os
import pathlib
from CoT.task import CoT_Task
from ToT.task import ToT_Task
from MCTS.task import MCTS_Task
import argparse
from utils.visualize import visualize
from utils.json_operator import *
from utils.verify_answer import *
from utils.self_consistency import get_consistency_output_scibench


def run(arguments):
    print('-'*30, 'Begin testing', '-'*30, '\n')
    file = f'data/{arguments.task_name}/{arguments.file}.json'
    try:
        data_list = read_json(file)
        data_len = len(data_list)
    except Exception as e:
        print(f'File must be standardized json!\nError type:{e}\n')
        return
    assert data_len > 0, "Data list is empty!\n"
    assert 'content' in data_list[0].keys() and 'answer' in data_list[0].keys(), "Key error, Make sure json object contain correct keys!\n"

    output_list = []
    correct_count = 0
    for i in range(data_len):
        # solve
        print(f'Begin to solve the problem {i+1}...\n')
        data = data_list[i]['content']
        answer = data_list[i]['answer']
        if arguments.mode == 'cot':
            Task = CoT_Task(data, arguments.propose_method, arguments.value_method, arguments.temperature, evaluate=arguments.evaluate)
            if arguments.consistency:
                outputs = []
                for cnt in range(3):
                    output = Task.run()
                    outputs.append(output)
                output = get_consistency_output_scibench(outputs)
            else:
                output = Task.run()

        elif arguments.mode == 'tot':
            Task = ToT_Task(data, arguments.propose_method, arguments.value_method, arguments.algorithm,
                            arguments.branch, arguments.select_branch, arguments.max_depth, arguments.end_gate,
                            arguments.select_method, arguments.temperature, use_case_prompt=arguments.use_case_prompt,
                            low=arguments.low, high=arguments.high, evaluate=arguments.evaluate)
            output, root = Task.run()
            if arguments.visualize:
                visualize(root, Task, arguments.task_name, arguments.file, i + 1)
        else:
            Task = MCTS_Task(data, arguments.propose_method, arguments.value_method, arguments.branch, arguments.end_gate,
                             arguments.roll_policy, arguments.roll_branch, arguments.roll_forward_steps, arguments.time_limit,
                             arguments.iteration_limit, arguments.exploration_constant, arguments.alpha, arguments.inf,
                             arguments.temperature, use_case_prompt=arguments.use_case_prompt, use_reflection=arguments.use_reflection,
                             low=arguments.low, high=arguments.high, evaluate=arguments.evaluate)
            output, root = Task.run()
            if arguments.visualize:
                visualize(root, Task, arguments.task_name, arguments.file, i + 1)

        # evaluate metrics
        if arguments.evaluate:
            result = verify_float(answer, output['summary'])
            output.update({'answer': answer, 'accurate': result})
            if result:
                print(f'The answer of problem {i+1} is correct.\n')
                correct_count += 1
            else:
                print(f'The answer of problem {i+1} is wrong.\n')
        print(f'The solution to problem {i+1} is complete.\n')

        # output
        base_dir = os.getcwd()
        output_dir = pathlib.Path(f'{base_dir}/outputs/{arguments.task_name}/{arguments.file}/{Task.mode}')
        output_file = f'{base_dir}/outputs/{arguments.task_name}/{arguments.file}/{Task.mode}/{Task.propose_method}_{Task.value_method}.json'
        output_list.append(output)
        pathlib.Path.mkdir(output_dir, exist_ok=True, parents=True)
        dump_json(output_file, output_list)

    print('_' * 60)
    # accuracy
    if args.evaluate:
        print(f'Test accuracy:{correct_count / data_len}\n')
        print(f'Correct number of problems:{correct_count}\nTotal number of questions:{data_len}\n')
    print('_' * 60)


def parse_args():
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--task_name', type=str, default='scibench')
    base_args.add_argument('--file', type=str, default='thermo_standardized')  # json
    base_args.add_argument('--propose_method', type=str, choices=['gpt', 'glm', 'llama', 'local'], default='glm')
    base_args.add_argument('--value_method', type=str, choices=['gpt', 'glm', 'local'], default='local')
    base_args.add_argument('--mode', type=str, choices=['cot', 'tot', 'mcts'], default='tot')
    base_args.add_argument('--temperature', type=float, default=0.7)
    base_args.add_argument('--time_limit', type=int, default=None)
    base_args.add_argument('--iteration_limit', type=int, default=100)
    base_args.add_argument('--roll_policy', type=str, choices=['random', 'greedy'], default='greedy')
    base_args.add_argument('--exploration_constant', type=float, default=0.4)
    base_args.add_argument('--roll_forward_steps', type=int, default=2)
    base_args.add_argument('--end_gate', type=float, default=0.9)  # End threshold
    base_args.add_argument('--branch', type=int, default=3)
    base_args.add_argument('--roll_branch', type=int, default=1)
    base_args.add_argument('--inf', type=float, default=0.8)
    base_args.add_argument('--evaluate', type=str, default='scibench')  # Whether to evaluate (empty means no evaluation)
    base_args.add_argument('--alpha', type=float, default=0.5)
    base_args.add_argument('--visualize', type=bool, default=False)  # visualization
    base_args.add_argument('--use_case_prompt', type=bool, default=False)  # Use sample prompts
    base_args.add_argument('--use_reflection', type=str, choices=['simple', 'common'], default='simple')  # Use reflective mode
    base_args.add_argument('--low', type=float, default=0)
    base_args.add_argument('--high', type=float, default=1)
    base_args.add_argument('--algorithm', type=str, choices=['dfs', 'bfs'], default='dfs')
    base_args.add_argument('--select_branch', type=int, default=2)
    base_args.add_argument('--max_depth', type=int, default=8)
    base_args.add_argument('--select_method', type=str, choices=['greedy', 'sample'], default='greedy')
    base_args.add_argument('--consistency', type=bool, default=True)

    arguments = base_args.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    run(args)
```
MCTS_task.run()
```python

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
        self.mode = 'mcts' self.temperature = temperature self.max_tokens = max_tokens self.seed = seed
        self.max_length = max_length self.truncation = truncation self.do_sample = do_sample self.max_new_tokens = max_new_tokens
        self.branch = branch self.use_case_prompt = use_case_prompt self.low = low self.high = high
        self.evaluate = evaluate self.end_gate = end_gate self.use_reflection = use_reflection self.roll_policy = roll_policy
        self.roll_branch = roll_branch self.time_limit = time_limit self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant self.roll_forward_steps = roll_forward_steps self.alpha = alpha
        self.limit_type = None self.INF = inf self.node_count = 1 self.sample_value = sample_value self.answer = answer
        self.verify_method = verify_method self.reward_model_type = 'prm' if USE_PRM else 'vm' self.lang = lang
        self.weighted_verify = weighted_verify
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
#---------------------------
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
#---------------------------------------------------------
                    final_answer = {'content': self.question, 'solution': solution, 'summary': summary,
                                    'finish': finish}
                    if self.sample_value == 'simple':
                        node.trace_route()
                        new_value_samples = node.get_new_value_samples()
#-------------------------------------------------------
    def get_new_value_samples(self):  # get value samples from search tree (start from terminal node)
        if self.depth == 0:
            return []
        step_value = 1.0 / self.depth
        new_samples = []
        cur_node = self.parent
        while cur_node is not None:
            for child in cur_node.children.values():
                if child.on_final_route:
                    child_value = step_value * child.depth
                    new_item = {'steps': child.y, 'value': child_value}
                    new_samples.append(new_item)
                else:
                    child_value = max(step_value * (cur_node.depth - 1), 0)
                    new_item = {'steps': child.y, 'value': child_value}
                    new_samples.append(new_item)
            cur_node = cur_node.parent
        return new_samples
#------------------------------------------------
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
#----------------------- MCTS_task.verify_end_nodes
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
#--------------------------
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
#--------------------------------------------- MCTS_task.get_final_solution()
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
#------------------------------------------
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
```
MCTS함수
```python
def MCTS(mcts_task):
    root, node, finish = MCTS_search(mcts_task)

    if mcts_task.sample_value == 'full':
        print('采样完成。\n') #샘플링 완료.
        return None, -1, root
    else:
        if mcts_task.reward_model_type == 'vm':
            if finish is not None: 
                print(f'최종 해를 찾았습니다!\nSolution:{node.y}\n') #已找到最终解
                return node, finish, root

            else:
                best_node, best_V = root.getBestV()
                print(f'지정된 시간/횟수 내에 요구하는 가치의 해를 찾지 못하여, 최고 가치의 해를 대신 사용합니다。\nSolution:{best_node.y}\n') #在规定时间/轮次内未找到满足要求价值的解答，采用最高价值价值解答代替
                return best_node, -1, root
        else:
            print('해답 선택은 아직 지원되지 않습니다, 샘플링 종료。\n') #尚未支持解答选择，采样结束
            return None, -1, root

```
MCTS_search 함수
```python
def MCTS_search(mcts_task):
    root = treeNode('')

    if mcts_task.limit_type == 'time':
        timeLimit = time.time() + mcts_task.time_limit / 1000
        time_start = time.time()
        while time.time() < timeLimit:
            print(f'<새로운 탐색 라운드 시작, 현재 총 시간:{time.time() - time_start}>\n') #开始新搜索轮次，目前总时间
# ----------------------------------------            
def executeRound(root, mcts_task):
    # execute a selection-expansion-simulation-backpropagation round

    print('-' * 40)
    print('selection 단계\n') #选择节点阶段
# def selectNode(node, mcts_task):
#     while node.isFullyExpanded:
 def getBestChild(node, mcts_task)
 bestValue = mcts_task.low
     bestNodes = []
     for child in node.children.values():
         nodeValue = child.V + mcts_task.exploration_constant * math.sqrt(
             2 * math.log(node.numVisits) / child.numVisits) if child.numVisits > 0 else child.V + mcts_task.INF <--UCT
         if nodeValue > bestValue:
             bestValue = nodeValue
             bestNodes = [child]
         elif nodeValue == bestValue:
             bestNodes.append(child)
     return random.choice(bestNodes)             
#         node = getBestChild(node, mcts_task)
   
#     if isTerminal(node, mcts_task):
#         node.final_ans_flag = 1
#         return True, node
#     else:
#         return False, node    
    flag, node = selectNode(root, mcts_task)
    #terminal node임으로 True return
    if flag: 
        if mcts_task.sample_value != 'full':
            return True, node, root
        #이경우에는 계속 진행되지만 밑의 대부분 과정 pass
        else:
            node.reflection = '<end>'

    print('-' * 40)
    print('扩充阶段\n') #확장 단계
    if node.reflection == '<end>':
        print('이 단계를 건너 뜁니다。\n') #跳过此阶段
    else:
#--------------------------------- expand 함수 설명
def expand(node: treeNode, mcts_task):
    if not node.reflection:
        if mcts_task.use_reflection == 'common':
            reflection = mcts_task.get_reflection(node.y, node.depth + 1)
        else:  # simple이 기본 default
            reflection = mcts_task.get_simple_reflection(node.y, node.depth + 1)
#--------------------------실제 해결과정 solution을 생성하는 파트 
#---------------------------------------------------------------
    def get_simple_reflection(self, y, step_n):
        if step_n == 1:
            return '<continue>'
        if self.propose_method in ['local', 'mistral', 'llama'] and self.lang == 'en':
            if 'answer is' in y or '\\boxed' in y:
                return '<end>'
        if self.propose_method == 'mistral':
            reflection_prompt = self.single_reflection_wrap_simple_mistral(self.question, y, step_n)
# single_reflection_prompt_simple_mistral = ''' Given a science problem and some corresponding steps, if the given steps have already solved the problem and provided the final answer to the question, then you should output: "solved". Otherwise, please output: "unsolved". Following the instruction, output "unsolved" or "solved", with no other information. Problem: '''        
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
            return '<end>' #밑의 표준화과정 더 있으나 길어서 짤
#----------------------------------------------
#----------------------------------------------
        node.update_reflection(reflection)
    if node.reflection == '<end>':
        return node
    actions = get_next_steps_expand(node, mcts_task)
#------------------------------------------- get_next_steps_expand함수
#------------------------------------------------------------
def get_next_steps_expand(node: treeNode, mcts_task):
    next_steps = []
    reflection = node.reflection
    for i in range(mcts_task.branch): #디폴트 값: 3
        proposal = ''
        cnt = 3
        while not proposal and cnt:
            if mcts_task.use_reflection == 'common':
                proposal = mcts_task.get_next_step_use_reflection(node.y, node.depth + 1, reflection)
            else:
                proposal = mcts_task.get_next_step(node.y, node.depth + 1)
#------------------------------------get_next_step 설명
    # def get_next_step(self, y, step_n):
    #     if self.use_case_prompt:
    #         prompt = self.single_propose_prompt_wrap(self.question, y, step_n)
    #     else:
    #         if self.propose_method == 'gpt':
    #             prompt = self.zero_single_propose_wrap_gpt(self.question, y, step_n, self.lang)
    #         elif self.propose_method == 'mistral' or self.propose_method == 'llama':
    #             prompt = self.zero_single_propose_wrap_mistral(self.question, y, step_n)
    #         else:
    #             prompt = self.zero_single_propose_wrap(self.question, y, step_n, self.lang)

    #     response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
    #                             self.max_length,
    #                             self.truncation, self.do_sample, self.max_new_tokens)
    #     p = ''
    #     for _ in response:
    #         p = p + _ + ' '
    #     p = p.strip()
    #     #strip함수 "  Hello, World!  " -> "Hello, World!"
    #     if self.lang == 'zh':
    #         if '下一步:' in p: #다음단계
    #             stp = p.split('下一步:')[1].strip()  #s_{i+1} 도출

    #             revised_ = '步骤' + str(step_n) + ':' + stp
    #             print(f'标准化后新的步骤:{revised_}\n') # 표준화(일관된 형식으로 정리)한 후, 그에 따라 새롭게 정리된 단계(스텝)
    #             return revised_ + '\n'
#---------------------------------------------------- get_next_steps 함수끝
            cnt -= 1
        if not proposal:
            continue
        next_steps.append(proposal)
    return next_steps
#------------------------------------------------- get_next_steps_expand함수 설명 끝
#------------------------------------------------
    if not actions:
        node.update_reflection('<end>')
        return node

    for action in actions:
        if action not in node.children.keys():
            node.append_children(action)
#------------------------------------------------------
    def append_children(self, new_pcd: str):
        node = treeNode(new_pcd, self, self.depth + 1)
        node.update_y_from_parent() # self.y = self.pcd or self.y = self.parent.y + self.pcd
        self.children.update({new_pcd: node})
        return self
#------------------------------------------------------
            child = node.children[action]
            value = mcts_task.get_step_value(child.y)
#----------------------------------------------------get step value 함수 설명
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
#------------------------------get value 함수 설명
#----------------------------------------------
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
#---------------------------------------------get value함수 설명끝
#---------------------------------------------
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
#----------------------------------------------------get step value 함수 설명 끝
            child.update_value(value) #이전 child.y의 value를 get step value 함수로 구함 = value
            if mcts_task.sample_value == 'full':
                if mcts_task.use_reflection == 'common':
                    child.update_reflection(mcts_task.get_reflection(child.y, child.depth + 1))
                else:
                    child.update_reflection(mcts_task.get_simple_reflection(child.y, child.depth + 1))
            child.visit_sequence = mcts_task.node_count
            mcts_task.update_count()
    node.isFullyExpanded = True
    return node
#--------------------------------- expand 함수끝
        node = expand(node, mcts_task)

    if mcts_task.reward_model_type == 'vm':
        print('-' * 40)
        print('모의 탐색 단계\n') #模拟搜索阶段
        if node.reflection == '<end>':
            print('이 단계를 건너 뜁니다。\n')
        else:
#----------------------------------- getBestChild 함수
# def getBestChild(node, mcts_task):
#     bestValue = mcts_task.low
#     bestNodes = []
#     for child in node.children.values():
#         nodeValue = child.V + mcts_task.exploration_constant * math.sqrt(
#             2 * math.log(node.numVisits) / child.numVisits) if child.numVisits > 0 else child.V + mcts_task.INF
#         if nodeValue > bestValue:
#             bestValue = nodeValue
#             bestNodes = [child]
#         elif nodeValue == bestValue:
#             bestNodes.append(child)
#     return random.choice(bestNodes)
#----------------------------------- getBestChild 함수 설명 끝            
            roll_node = getBestChild(node, mcts_task)
#--------------------------------------------greedypolicy 함수 설명
 def greedyPolicy(node: treeNode, mcts_task):
     max_V = mcts_task.low
     strs = node.y
     cur_step = node.depth + 1
     if mcts_task.use_reflection == 'common':
         reflection = mcts_task.get_reflection(strs, cur_step)
     else:
         reflection = mcts_task.get_simple_reflection(strs, cur_step)
     node.update_reflection(reflection)
     if reflection == '<end>':
         print('This step has been resolved and does not require simulation.\n')
         return node.V
     for i in range(mcts_task.roll_forward_steps):
# #------------------- get_next_steps_roll 함수
# def get_next_steps_roll(y: str, step_n: int, mcts_task):
#      next_steps = []
#      for i in range(mcts_task.roll_branch): #디폴트 roll_branch=1
#          proposal = ''
#          cnt = 3
#          while not proposal and cnt:
#              proposal = mcts_task.get_next_step(y, step_n) #다음단계 생성하는 함수
#              cnt -= 1
#          if not proposal:
#              continue
#          next_steps.append(proposal)
#      return next_steps
# #------------------------------------
         actions = get_next_steps_roll(strs, cur_step, mcts_task)  # str_list
         if not actions:
             break
         new_ys = [strs + action for action in actions]
         cur_step += 1
         values = [mcts_task.get_step_value(new_y) for new_y in new_ys]
         idx = numpy.argmax(values)
         strs = new_ys[idx]
         value = values[idx]
         if value > max_V:
             max_V = value
         if mcts_task.use_reflection == 'common':
             cur_ref = mcts_task.get_reflection(strs, cur_step)
         else:
             cur_ref = mcts_task.get_simple_reflection(strs, cur_step)
         if cur_ref == '<end>':
             break
     return max_V
#-------------------------------------------------greedy policy 함수 설명 끝         
            best_V = greedyPolicy(roll_node, mcts_task) if mcts_task.roll_policy == 'greedy' else randomPolicy(roll_node,
                                                                                                               mcts_task)
            roll_node.V = roll_node.V * (1 - mcts_task.alpha) + best_V * mcts_task.alpha
            roll_node.numVisits += 1

    print('-' * 40)
    print('역전파 단계\n') #反向传播阶段
# def back_propagate(node):
#     while node is not None:
#         node.numVisits += 1
#         if node.isFullyExpanded:
#             child_Vs = [child.V * child.numVisits for child in node.children.values()]
#             total_num_visits = sum([child.numVisits for child in node.children.values()])
#             if total_num_visits > 0:
#                 node.V = sum(child_Vs) / total_num_visits
#         node = node.parent
    
    back_propagate(node)
    return False, node, root
# --------------------------------------------- executeround 함수 끝            
            flag, node, root = executeRound(root, mcts_task)
            if flag:
                print('해결책을 찾았습니다！\n')#已找到解决方案
                return root, node, time.time() - time_start
    else:
        for i in range(mcts_task.iteration_limit):
            print(f'<새로운 탐색 라운드 시작, 현재 완료된 라운드 수:{i}>\n')
            flag, node, root = executeRound(root, mcts_task)
            if flag:
                print('해결책을 찾았습니다！\n')
                return root, node, i + 1
    return root, None, None
```
value prompt
```
critic_simplified = '''
你的任务是根据给定的理科问题和已有的解答步骤，判断这些步骤能否顺利解决该问题并输出分数。打分应该是0到1之间的小数，如果已有步骤全部不正确（每一步都错了）则是 0 分。如果已有步骤全部正确，且计算出了答案则是 1 分。已有步骤错的越多，分数越接近 0 分。已有步骤越接近最终答案，分数越接近 1 分。仅含有文字描述而没有计算式的步骤一般应该给分低，给大于或等于0.9分必须是已经计算出答案具体数值的（思路完整但没有计算出答案或者只列出了计算式的必须给低于0.9）。
先生成分析，后给出分数，你的分析和给分应该全部基于输入给定的步骤，不要继续生成下面的步骤。请学习以下样例。

输入:
问题: 讨论 p 取何值时, 广义积分$\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx$收敛。
已有步骤: 
步骤1: 要说明积分收敛，可以考虑将积分分成两部分：$$ \\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2} dx = \\int_0^1 \\frac{x^p \\ln x}{(1+x^2)^2} dx + \\int_1^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2} dx $$
步骤2: 对于第一部分，$0 \\leq \\frac{x^p \\ln x}{(1+x^2)^2} \\leq x^p$，因此它收敛当且仅当 $p>-2$。
输出:
分析: 第1步正确得到了拆分积分的思路，但第2步推导错误，对于收敛性的判断存在问题。$0 \\leq \\frac{x^p \\ln x}{(1+x^2)^2} \\leq x^p$，根据\\int_0^1 x^p dx收敛当且仅当$p>-1$，因此原积分收敛当且仅当 $p>-1$，而不是 $p>-2$。
分数: 0.1

输入:
问题: 求数列${n^{1/n}}$(n=1、2、3...为正整数)的最大项的值。
已有步骤:
步骤1: 考虑求导数：我们可以将数列 $n^{1/n}$ 视为函数 $f(x) = x^{1/x}$，然后求函数的导数 $f'(x)$。通过求导数，我们可以找到函数的增减性，进而确定数列的最大值所对应的正整数 $n$ 值。
输出:
分析: 已有步骤中的第1步是正确的，它建立了求解问题的基本思路，即将数列视为函数并通过求导数来分析函数的增减性。然而，这只是解题的一部分，还需要进一步的步骤来找到最大值所对应的正整数 $n$ 值以及求得最大值。因此，已有步骤还没有推断出答案。
分数: 0.2

输入:
问题: 求函数$f(x)=1+x^2$在区间$[-1,2]$上的平均值。
已有步骤:
步骤1: 利用定积分求解平均值：我们可以利用定积分来求解函数在区间 $[-1,2]$ 上的平均值。
步骤2: 首先，我们需要计算定积分 $\\int_{-1}^{2} (1+x^2) dx=6$。
步骤3: 然后，我们可以利用定积分的性质，将定积分的结果除以区间的长度，即 $\\frac{\\int_{-1}^{2} (1+x^2) dx}{3}$，这应该就是函数在区间上的平均值。
步骤4: 计算上面的式子，得到结果为$\\frac{\\int_{-1}^{2} (1+x^2) dx}{3}=\\frac{6}{3}=2$，因此函数的平均值为2。
输出:
分析: 所有步骤均推导正确，且已有步骤已经计算出答案为$2$，可以得到满分1分。
分数: 1

输入:
问题: 求数列${n^{1/n}}$(n=1、2、3...为正整数)的最大项的值。
已有步骤:
步骤1: 考虑求导数：我们可以将数列 $n^{1/n}$ 视为函数 $f(x) = x^{1/x}$，然后求函数的导数 $f'(x)$。通过求导数，我们可以找到函数的增减性，进而确定数列的最大值所对应的正整数 $n$ 值。
步骤2: 基于上一步的思路，对于函数 $f(x) = x^{1/x}$，我们可以取自然对数来简化求导过程，得到 $g(x)=\\ln(f(x)) = \\frac{1}{x}\\ln(x)$，再对g(x)求导数。
步骤3: 我们进行求导，$$\\frac{d}{dx}\\left(\\ln(f(x))\\right) = -\\frac{1}{x^2} \\ln(x) - \\frac{1}{x^2} + \\frac{1}{x^2} \\ln(x) = -\\frac{1}{x^2}$$。这个导数始终是负数，表示 $f(x)$ 在正整数范围内是递减的。、
输出:
分析: 前两步正确分析出了进行求导的思路，但第3步具体求导过程出错。求导的正确过程为：$$\\frac{d}{dx}\\left(\\ln(f(x))\\right) = -\\frac{1}{x^2} \\ln(x) + \\frac{1}{x^2}$$，而不是$-\\frac{1}{x^2}$。
分数: 0.2

输入:
问题: 求函数$f(x)=1+x^2$在区间$[-1,2]$上的平均值。
已有步骤:
步骤1: 考虑函数在区间端点处的值：我们可以计算函数在区间端点处 $x=-1$ 和 $x=2$ 的值，即 $f(-1)=1+(-1)^2=2$ 和 $f(2)=1+2^2=5$。
步骤2: 然后我们可以计算函数在这两个端点处的值的平均值，即 $\\frac{2+5}{2}=3.5$。这就是函数在区间 $[-1,2]$ 上的平均值。
输出:
分析: 全部推导步骤均错误，应该给0分。函数在区间上的平均值应该等于函数在区间上的积分除以区间长度，即$\\frac{\\int_{-1}^{2} (1+x^2) dx}{3}=2$，不能简单地认为其等于区间端点值函数值的平均值。
分数: 0

输入:
问题: 求数列${n^{1/n}}$(n=1、2、3...为正整数)的最大项的值。
已有步骤:
步骤1: 考虑求导数：我们可以将数列 $n^{1/n}$ 视为函数 $f(x) = x^{1/x}$，然后求函数的导数 $f'(x)$。通过求导数，我们可以找到函数的增减性，进而确定数列的最大值所对应的正整数 $n$ 值。
步骤2: 基于上一步的思路，对于函数 $f(x) = x^{1/x}$，我们可以取自然对数来简化求导过程，得到 $g(x)=\\ln(f(x)) = \\frac{1}{x}\\ln(x)$，再对g(x)求导数。
步骤3: 经过计算，我们得到 $g(x)$ 导数为 $$-\\frac{1}{x^2}\\ln(x) + \\frac{1}{x^2}$$
输出:
分析: 已有步骤均推导正确，但还没有具体计算出最大项的值，即没有计算出答案。还需要分析导数的正负性以了解$f(x)$的增减性。
分数: 0.6

输入:
问题: 讨论 p 取何值时, 广义积分$\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx$收敛。
已有步骤: 
步骤1: 记$J=\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_1=\\int_0^{1} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_2=\\int_1^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, 则广义积分$J$收敛当且仅当$J_1, J_2$都收敛。
步骤2: 当$x \\rightarrow 0^+$时，$\\frac{x^p \\ln x}{(1+x^2)^2} \\sim x^p \\ln x$ ，所以 $J_1$ 收敛当且仅当 $p > -1$。
步骤3: 当$x \\rightarrow +\\infty$时，$\\frac{x^p \\ln x}{(1+x^2)^2} \\sim \\frac{\\ln x}{x^{4-p}}$，所以 $J_2$ 收敛当且仅当 $p < 4$。
输出:
分析: 前两步正确，但第3步推导出错。当$x \\rightarrow +\\infty$时，$\\frac{x^p \\ln x}{(1+x^2)^2} \\sim \\frac{\\ln x}{x^{4-p}}$，根据\\int_0^{+\\infty} x^m dx收敛当且仅当$m<-1$，因此原积分收敛当且仅当 $p-4 < -1$，即$p < 3$，而不是 $p < 4$。
分数: 0.2

输入:
问题: 求函数$f(x)=-\\frac{1}{2}*(x^2)+2*x-1$在R上的最大值。
已有步骤:
步骤1: 求导数：我们可以求出函数$f(x)$的导数$f'(x)$，即$f'(x)=-x+2$。通过求导数，我们可以找到函数的增减性，进而确定函数在R上的最大值所对应的$x$值。
步骤2: 我们可以计算 $f'(x)$ 在 $x=1$ 时的值，即 $f'(1)=1$。由此可知，在 $x=1$ 处，函数 $f(x)$ 取得极大值，也就是最大值。
输出:
分析: 第一步正确，但第2步推导出错。计算$f'(x)$ 在 $x=1$ 时的值并不能告诉我们函数整体的增减性，没有意义。由 $f'(1)=1$ 不能推出函数在 $x=1$ 处，函数 $f(x)$ 取得极大值，极大值应该满足导数为0。
分数: 0.1

输入:
问题: 求数列${n^{1/n}}$(n=1、2、3...为正整数)的最大项的值。
已有步骤:
步骤1: 考虑求导数：我们可以将数列 $n^{1/n}$ 视为函数 $f(x) = x^{1/x}$，然后求函数的导数 $f'(x)$。通过求导数，我们可以找到函数的增减性，进而确定数列的最大值所对应的正整数 $n$ 值。
步骤2: 基于上一步的思路，对于函数 $f(x) = x^{1/x}$，我们可以取自然对数来简化求导过程，得到 $g(x)=\\ln(f(x)) = \\frac{1}{x}\\ln(x)$，再对g(x)求导数。
步骤3: 经过计算，我们得到 $g(x)$ 导数为 $$-\\frac{1}{x^2}\\ln(x) + \\frac{1}{x^2}$$
步骤4: 接着，我们可以分析导数值的正负性。这个导数在 $x > e$ 时是负数，而在 $x < e$ 时是正数。这意味着函数 $f(n)$ 在 $n > e$ 时是递减的，而在 $n < e$ 时是递增的。
输出:
分析: 已有步骤均推导正确，分析出了函数的增减性，但还没有具体计算出最大项的值，即没有计算出答案，所以不能给大于等于0.9的分数。但由于已有步骤已经很接近计算出答案，所以分数应该接近0.9分。
分数: 0.8

输入:
问题: 求函数$f(x)=x+1$与直线$x=0$，$x=1$和x轴围成的图形的面积。
已有步骤:
步骤1: 根据定积分的几何意义，求解函数的定积分即为所求图形的面积，可以直接将计算结果作为最终答案。
输出:
分析: 第1步分析是正确的，但是表达比较模糊，对解题的帮助非常小，更没有实际计算出答案，因此只能给很少的分数。更合适的表述为：根据定积分的几何意义，所求面积应该为$f(x)=x+1$在区间$[0,1]$上的定积分。
分数: 0.1

下面给定一个问题和已有的步骤，给出分析和打分。注意不要在分析中输出接下来的步骤，评分应该完全依据输入给定的步骤。
输出格式限定为:”分析:...\n分数:...“，其中...表示省略的输出内容，这是你需要填充的部分。

输入:
问题: '''

```
