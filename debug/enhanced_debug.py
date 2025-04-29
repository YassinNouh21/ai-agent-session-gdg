"""
Enhanced debug functions for the LangGraph code assistant.

This module provides enhanced debugging versions of the core functions
used in the LangGraph code assistant workflow.
"""
from dotenv import load_dotenv

load_dotenv(override=True)

def enhanced_generate(state, code_gen_chain=None, concatenated_content=None):
    """
    Generate a code solution with enhanced debug prints
    
    Args:
        state: The current state dictionary
        code_gen_chain: The chain used for code generation
        concatenated_content: The context content for code generation
    """
    print("="*50)
    print(f"📝 GENERATING CODE SOLUTION (Iteration {state['iterations']+1})")
    print("-"*50)

    # Extract state variables
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    # Log state information
    print(f"🔄 Current iteration: {iterations}")
    print(f"❌ Previous error state: {error}")
    print("-"*50)
    
    # Add correction prompt if there was an error
    if error == "yes":
        print("🔧 Adding correction prompt due to previous error")
        messages += [
            (
                "user",
                "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:"
            )
        ]

    # Generate code solution
    print("⚙️ Invoking code generation chain...")
    if code_gen_chain is None or concatenated_content is None:
        raise ValueError("code_gen_chain and concatenated_content must be provided")
        
    # Extract the question from messages
    question = messages[0][1] if messages else "Write a hello world function"
    
    code_solution = code_gen_chain.invoke({
        "context": concatenated_content, 
        "question": question
    })
    print("✅ Code solution generated successfully")
    
    # Add solution to messages
    print("📋 Adding solution to message history")
    messages += [
        (
            "assistant",
            f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}"
        )
    ]

    # Increment iterations counter
    iterations = iterations + 1
    print(f"🔢 Iteration counter incremented to: {iterations}")
    print("="*50)
    
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


def enhanced_code_check(state):
    """
    Check code for errors in imports and execution with enhanced debug prints
    """
    print("\n" + "="*80)
    print(f"🔍 CODE VALIDATION PROCESS (Iteration {state['iterations']})")
    print("="*80)

    # Extract state variables
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    # Get solution components
    imports = code_solution.imports
    code = code_solution.code
    
    # Print imports section
    print("\n📦 IMPORTS SECTION")
    print("╔" + "═"*78 + "╗")
    for line in imports.splitlines():
        # Highlight import keywords
        highlighted = line.replace("import ", "\033[93mimport\033[0m ")\
                         .replace("from ", "\033[93mfrom\033[0m ")\
                         .replace(" as ", "\033[93m as\033[0m ")
        print("║ " + highlighted + " "*(75-len(line)) + " ║")
    print("╚" + "═"*78 + "╝")
    
    # Print code section
    print("\n📄 CODE SECTION")
    print("╔" + "═"*78 + "╗")
    for line in code.splitlines():
        # Basic syntax highlighting
        highlighted = line
        # Highlight function definitions
        if "def " in line:
            highlighted = line.replace("def ", "\033[94mdef\033[0m ")
        # Highlight control flow keywords
        for keyword in ["if", "else", "for", "while", "try", "except", "return"]:
            if f"{keyword} " in line:
                highlighted = highlighted.replace(f"{keyword} ", f"\033[95m{keyword}\033[0m ")
        # Highlight string literals
        if '"' in line or "'" in line:
            highlighted = highlighted.replace('"', '\033[92m"\033[0m')\
                                   .replace("'", "\033[92m'\033[0m")
        print("║ " + highlighted + " "*(75-len(line)) + " ║")
    print("╚" + "═"*78 + "╝")

    # Check imports
    print("\n🧪 TESTING IMPORTS")
    print("┌" + "─"*78 + "┐")
    try:
        print("│ ⚙️  Executing imports...".ljust(79) + "│")
        exec(imports)
        print("│ ✅ Import test: PASSED".ljust(79) + "│")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print("│ ❌ Import test: FAILED".ljust(79) + "│")
        print("│ " + f"Error Type: {error_type}".ljust(77) + " │")
        print("│ " + f"Message: {error_msg}".ljust(77) + " │")
        print("│ " + "Details:".ljust(77) + " │")
        for line in repr(e).splitlines():
            print("│ " + f"  {line}".ljust(77) + " │")
        print("└" + "─"*78 + "┘")
        error_message = [("user", f"Your solution failed the import test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes"
        }
    print("└" + "─"*78 + "┘")

    # Check execution
    print("\n🧪 TESTING CODE EXECUTION")
    print("┌" + "─"*78 + "┐")
    try:
        print("│ ⚙️  Executing combined imports and code...".ljust(79) + "│")
        exec(imports + "\n" + code)
        print("│ ✅ Execution test: PASSED".ljust(79) + "│")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print("│ ❌ Execution test: FAILED".ljust(79) + "│")
        print("│ " + f"Error Type: {error_type}".ljust(77) + " │")
        print("│ " + f"Message: {error_msg}".ljust(77) + " │")
        print("│ " + "Traceback:".ljust(77) + " │")
        import traceback
        for line in traceback.format_exc().splitlines():
            print("│ " + f"  {line}".ljust(77) + " │")
        print("└" + "─"*78 + "┘")
        error_message = [("user", f"Your solution failed the code execution test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes"
        }
    print("└" + "─"*78 + "┘")

    # Success summary
    print("\n🎉 VALIDATION SUMMARY")
    print("┌" + "─"*78 + "┐")
    print("│ ✅ Import Test: PASSED".ljust(79) + "│")
    print("│ ✅ Execution Test: PASSED".ljust(79) + "│")
    print("│ 🎯 All validation checks completed successfully".ljust(79) + "│")
    print("└" + "─"*78 + "┘\n")
    
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no"
    }


def enhanced_reflect(state, code_gen_chain=None, concatenated_content=None):
    """
    Generate reflections on errors with enhanced debug prints
    
    Args:
        state: The current state dictionary
        code_gen_chain: The chain used for code generation
        concatenated_content: The context content for code generation
    """
    print("="*50)
    print(f"🤔 GENERATING REFLECTION (Iteration {state['iterations']})")
    print("-"*50)

    # Extract state variables
    messages = state["messages"]
    iterations = state["iterations"]
    code_solution = state["generation"]
    
    print(f"📋 Message history length: {len(messages)}")
    print(f"🔄 Current iteration: {iterations}")

    # Get the latest error message
    error_message = messages[-1][1] if messages and len(messages) > 1 else "Unknown error"
    
    # Generate reflections on the error
    print("⚙️ Generating reflections on error...")
    if code_gen_chain is None or concatenated_content is None:
        raise ValueError("code_gen_chain and concatenated_content must be provided")
    
    # Extract the original question/problem from the first user message
    question = messages[0][1] if messages else "Fix the code error"
        
    reflections = code_gen_chain.invoke({
        "context": concatenated_content, 
        "question": f"The following code had an error: {error_message}. Please analyze what went wrong and how to fix it."
    })
    print("✅ Reflections generated successfully")
    
    # Add reflections to messages
    print("📋 Adding reflections to message history")
    messages += [("assistant", f"Here are reflections on the error: {reflections}")]
    print("="*50)
    
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


def enhanced_decide_to_finish(state, max_iterations=3, flag="do not reflect"):
    """
    Determines whether to finish or continue iterating with enhanced debug prints
    
    Args:
        state: The current state dictionary
        max_iterations: Maximum number of iterations before stopping
        flag: Whether to use reflection ("reflect") or not ("do not reflect")
    """
    print("="*50)
    print("🔀 DECIDING NEXT STEP")
    print("-"*50)
    
    error = state["error"]
    iterations = state["iterations"]
    
    print(f"❌ Error state: {error}")
    print(f"🔄 Current iteration: {iterations}")
    print(f"🔢 Max iterations: {max_iterations}")
    print(f"🚩 Reflection flag: {flag}")

    # Finish if no errors or max iterations reached
    if error == "no" or iterations == max_iterations:
        if error == "no":
            print("✅ No errors detected, finishing")
        else:
            print(f"⚠️ Max iterations ({max_iterations}) reached, finishing despite errors")
        print("🏁 DECISION: FINISH")
        print("="*50)
        return "end"
    else:
        print("⚠️ Errors detected, retrying solution")
        print("🔄 DECISION: RE-TRY SOLUTION")
        
        # Choose whether to reflect or go straight to generation
        if flag == "reflect":
            print("🤔 Using reflection before next generation")
            print("="*50)
            return "reflect"
        else:
            print("⏩ Skipping reflection, going directly to generation")
            print("="*50)
            return "generate"

def setup_debug_workflow(code_gen_chain=None, concatenated_content=None, max_iterations=3, flag="do not reflect"):
    """
    Creates and returns a debug version of the workflow with enhanced prints
    
    Args:
        code_gen_chain: The chain used for code generation
        concatenated_content: The context content for code generation
        max_iterations: Maximum number of iterations before stopping
        flag: Whether to use reflection ("reflect") or not ("do not reflect")
    """
    # Import required dependencies directly from langgraph
    from langgraph.graph import StateGraph, START, END
    from typing_extensions import TypedDict
    from functools import partial

    # Define GraphState class
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            error : Binary flag for control flow to indicate whether test error was tripped
            messages : With user question, error messages, reasoning
            generation : Code solution
            iterations : Number of tries
        """
        error: str
        messages: list
        generation: str
        iterations: int
    
    # Create a new workflow with enhanced debug prints
    debug_workflow = StateGraph(GraphState)

    # Create partial functions with the required parameters
    generate_with_params = partial(enhanced_generate, 
                                 code_gen_chain=code_gen_chain,
                                 concatenated_content=concatenated_content)
    
    reflect_with_params = partial(enhanced_reflect,
                                code_gen_chain=code_gen_chain,
                                concatenated_content=concatenated_content)
    
    decide_with_params = partial(enhanced_decide_to_finish,
                               max_iterations=max_iterations,
                               flag=flag)

    # Define the nodes with enhanced debug functions
    debug_workflow.add_node("generate", generate_with_params)    # Enhanced debug generation node
    debug_workflow.add_node("check_code", enhanced_code_check)  # Enhanced debug validation node
    debug_workflow.add_node("reflect", reflect_with_params)      # Enhanced debug reflection node

    # Build graph connections
    debug_workflow.add_edge(START, "generate")      # Start with generation
    debug_workflow.add_edge("generate", "check_code")  # Check generated code

    # Add conditional routing based on code check result
    debug_workflow.add_conditional_edges(
        "check_code",
        decide_with_params,
        {
            "end": END,            # Success or max iterations reached
            "reflect": "reflect",  # Analyze error and reflect
            "generate": "generate"  # Try again without reflection
        }
    )

    # Connect reflection back to generation
    debug_workflow.add_edge("reflect", "generate")

    # Compile the debug workflow
    debug_app = debug_workflow.compile()
    print("✅ Debug workflow compiled successfully!")
    
    return debug_app


if __name__ == "__main__":
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field
    import os
    import sys
    
    # Ensure the debug directory is in the Python path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    # Define code schema
    class code(BaseModel):
        """Schema for code solutions."""
        prefix: str = Field(description="Description of the problem and approach")
        imports: str = Field(description="Code block import statements")
        code: str = Field(description="Code block not including import statements")

    # Setup chain with intentionally low temperature to demonstrate iteration
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a coding assistant with expertise in Python programming.
            
            Here is a full set of documentation:
            -------
            {context}
            -------
            
            Answer the user question based on the above provided documentation. 
            Ensure any code you provide can be executed with all required imports and variables defined.
            Structure your answer with a description of the code solution.
            Then list the imports. And finally list the functioning code block.
            
            IMPORTANT: All code must be in Python. Do not use any other programming language.
            Focus on writing functional code that addresses the main requirements first.
            
            Here is the user question:"""
        ),
        ("user", "{question}")
    ])
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    code_gen_chain = prompt | llm.with_structured_output(code)
    
    # Example documentation context with task to find and use weather function
    concatenated_content = """
    The weather data function can be found in debug/my_code.py. 
    
    The function is named get_weather_data and takes the following parameters:
    - cities: list of city names to get weather data for
    - api_key: optional API key parameter (has a default value)
    
    The function returns a dictionary with weather information including:
    - average_temperature
    - hottest_city
    - coldest_city
    
    Your task is to:
    1. Import this function correctly from debug/my_code.py
    2. Call it with the cities London, Paris, and Tokyo
    3. Extract and print the hottest city, coldest city, and average temperature
    4. Handle any potential errors
    """
    
    # Complex problem that requires finding and using existing code
    complex_problem = """
    Use the weather data function from debug/my_code.py to:
    1. Get weather data for London, Paris, and Tokyo
    2. Print out which city is hottest, which is coldest, and the average temperature
    3. Handle any potential errors that might occur
    
    The function is already implemented in debug/my_code.py - you need to correctly
    import and use it, not reimplement it.
    """
    
    # Initialize the debug workflow with multiple iterations
    debug_workflow = setup_debug_workflow(
        code_gen_chain=code_gen_chain,
        concatenated_content=concatenated_content,
        max_iterations=5,  # Allow more iterations
        flag="reflect"     # Enable reflection between attempts
    )
    
    # Initial state
    state = {
        "messages": [("user", complex_problem)],
        "iterations": 0,
        "error": "no",
        "generation": None
    }
    
    # Run the debug workflow using invoke()
    print("\n🚀 Starting multi-iteration workflow to find and use weather function...\n")
    result = debug_workflow.invoke(state)
    
    # Print final summary
    print("\n📊 Multi-iteration Workflow Summary:")
    print(f"Total iterations: {result['iterations']}")
    print(f"Final error state: {result['error']}")
    print(f"Message history length: {len(result['messages'])}")
    
    # Print the final solution
    if result['generation']:
        print("\n🏆 Final Solution:")
        print("\nPrefix:")
        print(result['generation'].prefix)
        print("\nImports:")
        print(result['generation'].imports)
        print("\nCode:")
        print(result['generation'].code)