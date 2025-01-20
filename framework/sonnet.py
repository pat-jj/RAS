from claude_api import get_claude_response



def text_to_triples_sonnet(text: str) -> str:
    """
    Generates relationship triples from input text.
    
    Args:
        text (str): Input text to extract relationships from
        
    Returns:
        str: Comma-separated triples in format (S> subject| P> predicate| O> object)
    """
    prompt = f"""Extract relationship triples from the given text. Each triple should have exactly one subject (S>), one predicate (P>), and one object (O>).

Rules:
1. Extract as many meaningful triples as possible
2. Each triple must be in format: (S> subject| P> predicate| O> object)
3. Multiple triples should be separated by commas
4. Avoid using pronouns (it/he/she) - always use the actual names
5. Keep all entities in their original case (uppercase/lowercase)
6. Make predicates clear and specific
7. When input is only an entity, just output the entity itself
8. [IMPORTANT] Do not include any other text in the output, only the triples or the entity (for Rule 7 case).

Example Input:
"William Gerald Standridge (November 27, 1953 – April 12, 2014) was an American stock car racing driver. He was a competitor in the NASCAR Winston Cup Series and Busch Series."

Example Output:
(S> William gerald standridge| P> Nationality| O> American),
(S> William gerald standridge| P> Occupation| O> Stock car racing driver),
(S> William gerald standridge| P> Competitor| O> Busch series),
(S> William gerald standridge| P> Competitor| O> Nascar winston cup series),
(S> William gerald standridge| P> Birth date| O> November 27, 1953),
(S> William gerald standridge| P> Death date| O> April 12, 2014)

Input Text: 
{text}

"""

    # Here you would add your actual implementation to get response from Claude API
    # For example:
    response = get_claude_response(llm="sonnet", prompt=prompt)
    return response.strip()



def planner_sonnet(planner_intput):
    planner_instruction = """You are a planner to determine if the question can be answered with current information (Subquery [PREV_SUBQ] and retrieved graph information [PREV_GRAPH_INFO]) and output the appropriate label as well as the subquery if needed.
Output [NO_RETRIEVAL] if the question can be directly answered with the question itself without any retrieval. You are expected to output [NO_RETRIEVAL] either if you believe an LLM is knowledgeable enough to answer the question, or if you believe the question type is not suitable for retrieval.
Output [SUBQ] with an subquery for retrieval if still needs a subquery. Do not make an similar subquery that has been made before ([PREV_SUBQ]), as it is very likely to retrieve the same information.
Output [SUFFICIENT] if the question can be answered with the provided information.
The main question starts with "Question: ".
"""

    planner_prompt_with_examples = f"""
Examples:
------------------------------------------------------------------------------------------------
Example 1 (Use the information in [PREV_GRAPH_INFO] to further generate a new subquery):
Input: 
[PREV_SUBQ] Where is The Pick Motor Company Limited located?
[PREV_GRAPH_INFO] ['(S> Pick Motor Company Limited| P> Alias| O> New Pick Motor Company)', '(S> Pick Motor Company Limited| P> Location| O> Stamford, Lincolnshire)', '(S> Pick Motor Company Limited| P> Operational period| O> 1899-1925)', '(S> Pick Motor Company Limited| P> Industry| O> Motor vehicle manufacturing)’]
Question: The Pick Motor Company Limited is located in a town on which river ?,

Output:
[SUBQ] Which river is Stamford located on?

------------------------------------------------------------------------------------------------
Example 2 (No relevant information found in [PREV_GRAPH_INFO]):
Input: 
[PREV_SUBQ] What medals did Michael Johnson win in the 1996 Olympics?
[PREV_GRAPH_INFO] ['(S> Michael Johnson| P> Nationality| O> American)', '(S> Michael Johnson| P> Birth date| O> September 13, 1967)', '(S> Michael Johnson| P> Sport| O> Track and field)', '(S> Michael Johnson| P> Team| O> United States Olympic team)']
Question: What was Michael Johnson's winning time in the 400m at the 1996 Olympics?

Output:
[SUBQ] What records or times did Michael Johnson set in the 400m at the 1996 Olympic Games?

------------------------------------------------------------------------------------------------
Example 3 (The current information is sufficient to answer the question):
Input: 
[PREV_SUBQ] What gaming control board in Ohio is Martin R. Hoke a member of?
[PREV_GRAPH_INFO] ['(S> Martin R. Hoke| P> Former position| O> Member of the United States House of Representatives)', '(S> Martin R. Hoke| P> Birth date| O> May 18, 1952)', '(S> Martin R. Hoke| P> Occupation| O> Politician)', '(S> Martin R. Hoke| P> State| O> Ohio)', '(S> Martin R. Hoke| P> Nationality| O> American)', '(S> Martin R. Hoke| P> Party| O> Republican)', '(S> Martin R. Hoke| P> Member of| O> Ohio Casino Control Commission)', '(S> Martin R. Hoke| P> Born| O> 1952)’] 
[PREV_SUBQ] What gaming control board provides oversight of Ohio's casinos?
[PREV_GRAPH_INFO] [\"(S> Ohio Casino Control Commission| P> Function| O> Provides oversight of the state's casinos)\", '(S> Ohio Casino Control Commission| P> Location| O> Ohio)', '(S> Ohio Casino Control Commission| P> Type| O> Gaming control board)', '(S> Ohio Casino Control Commission| P> Abbreviation| O> OCCC)’]
Question: Martin R. Hoke, is an American Republican politician, member of which gaming control board in Ohio that provides oversight of the state's casinos?


Output:
[SUFFICIENT]

------------------------------------------------------------------------------------------------
Example 4 (The question is not suitable for retrieval):
Input:
Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response.
##Input:
I love pizza. While it's basically just cheese and bread you can top a pizza with vegetables, meat etc. You can even make it without cheese!
Pizza is the greatest food ever! I like the New York style.
I do too. I like that the crust is only thick and crisp at the edge, but soft and thin in the middle so its toppings can be folded in half. 
Absolutely! I am not that big of a fan of Chicago deep dish though

Output:
[NO_RETRIEVAL]

------------------------------------------------------------------------------------------------
Example 5 (You are knowledgeable enough to answer the question):
Input:
What is the capital of the United States?

Output:
[NO_RETRIEVAL]

------------------------------------------------------------------------------------------------
[VERY IMPORTANT] Please only either output (1) [NO_RETRIEVAL] or (2) [SUBQ] with an concrete subquery for retrieval, or (3) [SUFFICIENT] if the question can be answered with the provided information.
[VERY IMPORTANT] Do not output any other text. DO NOT make an identical subquery [SUBQ] that has been made before ([PREV_SUBQ])!
Now, your turn:
Input:
{planner_intput}

Output:
"""

    
    planner_input = planner_instruction + "\n" + planner_prompt_with_examples
    
    return get_claude_response(llm="sonnet", prompt=planner_input, max_tokens=200)



def answerer_sonnet(answerer_input, max_answer_length=100):
    answerer_instruction = """You are a answerer given a question and retrieved graph information.
Each [SUBQ] is a subquery we generated through reasoning for the question. The retrieved graph information follows each [SUBQ] is relevant graph information we retrieved to answer the subquery.
The main question starts with "Question: ". Please answer the question, with subqueries and retrieved graph information if they are helpful (do not use them if they are not helpful).
You must answer the question, even if there's no enough information to answer the question, or you are not sure about the answer.
"""
    
    
    answerer_prompt_with_examples = f"""
Examples:
------------------------------------------------------------------------------------------------
Example 1:
Input:
Question: Which person won the Nobel Prize in Literature in 1961, Ivo Andri or Nicholas Pileggi?",

Output:
Ivo Andri

------------------------------------------------------------------------------------------------
Example 2:
[SUBQ] What type of athlete is Darold Williamson?
Retrieved Graph Information: ['(S> Darold Williamson| P> Nationality| O> American)', '(S> Darold Williamson| P> Birth date| O> February 19, 1983)', '(S> Darold Williamson| P> Occupation| O> Track athlete)’]
[SUBQ] What specific skills are included in track and field events?
Retrieved Graph Information: ['(S> Athletics| P> Includes| O> Road running)', '(S> Track and field| P> Includes| O> Throwing)', '(S> Track and field| P> Includes| O> Jumping)', '(S> Athletics| P> Includes| O> Cross country running)', '(S> Track and field| P> Categorised under| O> Athletics)', '(S> Track and field| P> Includes| O> Running)', '(S> Track and field| P> Venue| O> Stadium with an oval running track enclosing a grass field)', '(S> Athletics| P> Includes| O> Track and field)', '(S> Athletics| P> Includes| O> Race walking)', '(S> Track and field| P> Based on skills of| O> Running)', '(S> Track and field| P> Based on skills of| O> Jumping)', '(S> Track and field| P> Based on skills of| O> Throwing)’]
Question: Darold Williamson is an athlete in what running and jumping sport?",

Output:
Track and field

------------------------------------------------------------------------------------------------
[VERY IMPORTANT] Please only output the answer to the question.
[VERY IMPORTANT] Do not output any other text.
Now, your turn:
Input:
{answerer_input}

Output:
"""

    answerer_input = answerer_instruction + "\n" + answerer_prompt_with_examples
    
    return get_claude_response(llm="sonnet", prompt=answerer_input, max_tokens=max_answer_length)