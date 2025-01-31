# multimodal-MCTS
{
  "content": "Calculate the sum of the first 10 prime numbers.",
  "answer": "129"
}
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
MCTS_task.run()
```python
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
# # bestValue = mcts_task.low
# #     bestNodes = []
# #     for child in node.children.values():
# #         nodeValue = child.V + mcts_task.exploration_constant * math.sqrt(
# #             2 * math.log(node.numVisits) / child.numVisits) if child.numVisits > 0 else child.V + mcts_task.INF <--UCT
# #         if nodeValue > bestValue:
# #             bestValue = nodeValue
# #             bestNodes = [child]
# #         elif nodeValue == bestValue:
# #             bestNodes.append(child)
# #     return random.choice(bestNodes)             
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

# def expand(node: treeNode, mcts_task):
#     if not node.reflection:
#         if mcts_task.use_reflection == 'common':
#             reflection = mcts_task.get_reflection(node.y, node.depth + 1)
#         else:  # simple
#             reflection = mcts_task.get_simple_reflection(node.y, node.depth + 1)
#         node.update_reflection(reflection)
#     if node.reflection == '<end>':
#         return node
#     actions = get_next_steps_expand(node, mcts_task)
#     if not actions:
#         node.update_reflection('<end>')
#         return node

#     for action in actions:
#         if action not in node.children.keys():
#             node.append_children(action)
#             child = node.children[action]
#             value = mcts_task.get_step_value(child.y)
#             child.update_value(value)
#             if mcts_task.sample_value == 'full':
#                 if mcts_task.use_reflection == 'common':
#                     child.update_reflection(mcts_task.get_reflection(child.y, child.depth + 1))
#                 else:
#                     child.update_reflection(mcts_task.get_simple_reflection(child.y, child.depth + 1))
#             child.visit_sequence = mcts_task.node_count
#             mcts_task.update_count()
#     node.isFullyExpanded = True
#     return node    
        node = expand(node, mcts_task)

    if mcts_task.reward_model_type == 'vm':
        print('-' * 40)
        print('모의 탐색 단계\n') #模拟搜索阶段
        if node.reflection == '<end>':
            print('이 단계를 건너 뜁니다。\n')
#         else:
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
            roll_node = getBestChild(node, mcts_task)
# def greedyPolicy(node: treeNode, mcts_task):
#     max_V = mcts_task.low
#     strs = node.y
#     cur_step = node.depth + 1
#     if mcts_task.use_reflection == 'common':
#         reflection = mcts_task.get_reflection(strs, cur_step)
#     else:
#         reflection = mcts_task.get_simple_reflection(strs, cur_step)
#     node.update_reflection(reflection)
#     if reflection == '<end>':
#         print('This step has been resolved and does not require simulation.\n')
#         return node.V
#     for i in range(mcts_task.roll_forward_steps):
# # def get_next_steps_roll(y: str, step_n: int, mcts_task):
# #     next_steps = []
# #     for i in range(mcts_task.roll_branch):
# #         proposal = ''
# #         cnt = 3
# #         while not proposal and cnt:
# #             proposal = mcts_task.get_next_step(y, step_n)
# #             cnt -= 1
# #         if not proposal:
# #             continue
# #         next_steps.append(proposal)
# #     return next_steps        
#         actions = get_next_steps_roll(strs, cur_step, mcts_task)  # str_list
#         if not actions:
#             break
#         new_ys = [strs + action for action in actions]
#         cur_step += 1
#         values = [mcts_task.get_step_value(new_y) for new_y in new_ys]
#         idx = numpy.argmax(values)
#         strs = new_ys[idx]
#         value = values[idx]
#         if value > max_V:
#             max_V = value
#         if mcts_task.use_reflection == 'common':
#             cur_ref = mcts_task.get_reflection(strs, cur_step)
#         else:
#             cur_ref = mcts_task.get_simple_reflection(strs, cur_step)
#         if cur_ref == '<end>':
#             break
#     return max_V            
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
# ---------------------------------------------            
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
