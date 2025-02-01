import re
from functools import partial
import re
from typing import Optional
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
#llm_verify -> extract_solution -> grade_answer함수 요약
#------------------------------------
from get_proposal import *
def llm_verify(ans, real_ans, judge_model='gpt-4-1106-preview'):
    prompt = '下面将输入两段文字，第一段文字为某道理科题目的一个解答或答案（不一定正确），第二段是这道题目的标准答案。请判断第一段解答得到的答案与标准答案在数学意义上是否一致，并根据判断直接输出‘0’或’1‘，不需要输出任何别的信息。如果答案一致，请输出‘1’；否则，只要答案不匹配，或者第一个文段中没有明确指出答案也没有输出latex表达式，请输出‘0’；如果第一段解答与标准答案之间关系模糊，请输出‘0’。\n'
# Two pieces of text will be entered below.

# The first piece is a solution or an answer to a science-related problem (such as math or physics). This answer may not necessarily be correct.
# The second piece is the standard answer to the problem.

# Now, determine whether the answer obtained from the first solution is mathematically equivalent to the standard answer.
# Based on your judgment, output the result as follows:

# If the two answers are mathematically identical, output '1'.
# If the answers do not match, or if the first piece does not explicitly state an answer or does not contain a LaTeX expression, output '0'.ㅡ 5
# If the relationship between the first solution and the standard answer is ambiguous, output '0'.
# The output must be strictly '0' or '1' without any additional information.
    qry = prompt + '文段1:' + ans + '\n' + '文段2:' + real_ans + '\n输出:'
    lbl = ''
    cnt = 5
    while lbl == '' and cnt:
        out = ''
        try:
            chat_comp = openai.ChatCompletion.create(model=judge_model, messages=[{"role": "user", "content": qry}])
            out = chat_comp.choices[0].message.content[0]
        except Exception as e:
            print(f'Error:{e}\n')
        if out == '0' or out == '1':
            lbl = out
        else:
            cnt -= 1
    if not cnt:
        return 0
    return int(lbl)

def extract_summary_from_solution(solution: str):
    pattern = r"\\boxed\{(.*)\}"
    # r"The result is \boxed{x^2 + 3x - 7}."
    match = re.findall(pattern, solution)
    if match:
        summary = 'The final answer is ' + match[-1]
    elif '####' in solution:
        extracted = solution.split('####')[-1].strip()
        if len(extracted) > 1:
            if extracted[-1] == '.':
                extracted = extracted[:-1].strip()
            if len(extracted) > 1:
                if extracted[0] == ':':
                    extracted = extracted[1:].strip()
        summary = 'The final answer is ' + extracted
    elif 'The final answer is' in solution:
        extracted = solution.split('The final answer is')[-1].strip()
        if len(extracted) > 1:
            if extracted[-1] == '.':
                extracted = extracted[:-1].strip()
            if len(extracted) > 1:
                if extracted[0] == ':':
                    extracted = extracted[1:].strip()
        summary = 'The final answer is ' + extracted
    elif 'The answer is' in solution:
        extracted = solution.split('The answer is')[-1].strip()
        if len(extracted) > 1:
            if extracted[-1] == '.':
                extracted = extracted[:-1].strip()
            if len(extracted) > 1:
                if extracted[0] == ':':
                    extracted = extracted[1:].strip()
        summary = 'The final answer is ' + extracted
    elif 'final answer is' in solution:
        extracted = solution.split('final answer is')[-1].strip()
        if len(extracted) > 1:
            if extracted[-1] == '.':
                extracted = extracted[:-1].strip()
            if len(extracted) > 1:
                if extracted[0] == ':':
                    extracted = extracted[1:].strip()
        summary = 'The final answer is ' + extracted
    elif 'answer is' in solution:
        extracted = solution.split('answer is')[-1].strip()
        if len(extracted) > 1:
            if extracted[-1] == '.':
                extracted = extracted[:-1].strip()
            if len(extracted) > 1:
                if extracted[0] == ':':
                    extracted = extracted[1:].strip()
        summary = 'The final answer is ' + extracted
    else:
        summary = ''
    print('Extracted summary: ', summary, '\n')
    return summary
    
def exact_match_score(prediction, ground_truth):
    prediction = extract_answer(prediction)
    return grade_answer(prediction, ground_truth)



def extract_answer(prediction):
    pattern = r"The final answer is \$([^$]*)\$"
    match = re.findall(pattern, prediction)
    if match:
        # print("match1")
        answer = match[0]
    else:
        pattern2 = r"\$([^$]*)\$"
        # "이 방정식은 $E = mc^2$ 이고, 다른 예제는 $x^2 + y^2 = r^2$ 입니다." ->['E = mc^2', 'x^2 + y^2 = r^2']
        match2 = re.findall(pattern2, prediction)
        if match2:
            # print("match2")
            answer = match2[0]
        else:
            if 'answer is' in prediction:
                answer = prediction.split('answer is')[-1].strip()
                if len(answer) > 1:
                    if answer[-1] == '.':
                        answer = answer[:-1].strip()
                    if len(answer) > 1:
                        if answer[0] == ':':
                            answer = answer[1:].strip()
            else:
                pattern3 = r'-?[0-9]+\.?[0-9]*'
                match3 = re.findall(pattern3, prediction)
                #"숫자 예제: -42, 3.14, 0.5, 그리고 100." -> ['-42', '3.14', '0.5', '100']
                if match3:
                    # print("match3")
                    answer = match3[-1]
                else:
                    answer = ""
    return answer

def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = normalize_answer(ground_truth)
    given_answer_normalized_mathd = normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
            ground_truth_normalized[0] != given_normalized[0]
            or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        is_correct = True
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _is_float(ground_truth_elem) and _is_float(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                if _str_is_int(ground_truth_elem):
                    try:
                        is_correct = round(float(given_elem)) == int(ground_truth_elem)
                    except:
                        is_correct = False
                else:
                    ground_truth_elem = float(ground_truth_elem)
                    given_elem = float(given_elem)
                    eps = abs(ground_truth_elem) * 0.04
                    if ground_truth_elem - eps <= given_elem <= ground_truth_elem + eps:
                        is_correct = True
                    else:
                        is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct

#---------------------------------------------------------
#grade_answer 함수의 normalize answer
def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        # \text{}로 감싸진 내용을 찾는 정규식
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

#-------------------------------------
#grade answer 함수의 _normalize



def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr

  
def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))

def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False
      
def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step

#----------------------------------
# grade answer의 split_tuple함수


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
            len(expr) > 2
            and expr[0] in TUPLE_CHARS
            and expr[-1] in TUPLE_CHARS
            and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems

#---------------------
#grade answer의 are_equal_under_sympy 함수
def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if abs(simplified) <= 0.04 * sympy.simplify(ground_truth_normalized):
                are_equal = True
    except:
        pass
    return are_equal


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)

def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
                sympy_parser.standard_transformations
                + (sympy_parser.implicit_multiplication_application,)
        ),
    )
