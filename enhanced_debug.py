"""
Enhanced debug functions for the LangGraph code assistant.

This module provides enhanced debugging versions of the core functions
used in the LangGraph code assistant workflow.
"""

def enhanced_generate(state):
    """
    Generate a code solution with enhanced debug prints
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
    # Import required dependencies from the notebook scope
    from __main__ import code_gen_chain, concatenated_content
    
    code_solution = code_gen_chain.invoke({
        "context": concatenated_content, 
        "messages": messages
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


def enhanced_reflect(state):
    """
    Generate reflections on errors with enhanced debug prints
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

    # Generate reflections on the error
    print("⚙️ Generating reflections on error...")
    # Import required dependencies from the notebook scope
    from __main__ import code_gen_chain, concatenated_content
    
    reflections = code_gen_chain.invoke({
        "context": concatenated_content, 
        "messages": messages
    })
    print("✅ Reflections generated successfully")
    
    # Add reflections to messages
    print("📋 Adding reflections to message history")
    messages += [("assistant", f"Here are reflections on the error: {reflections}")]
    print("="*50)
    
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


def enhanced_decide_to_finish(state):
    """
    Determines whether to finish or continue iterating with enhanced debug prints
    """
    print("="*50)
    print("🔀 DECIDING NEXT STEP")
    print("-"*50)
    
    error = state["error"]
    iterations = state["iterations"]
    
    # Import required dependencies from the notebook scope
    from __main__ import max_iterations, flag
    
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

def setup_debug_workflow():
    """
    Creates and returns a debug version of the workflow with enhanced prints
    """
    # Import required dependencies from the notebook scope
    from __main__ import StateGraph, GraphState, START, END
    
    # Create a new workflow with enhanced debug prints
    debug_workflow = StateGraph(GraphState)

    # Define the nodes with enhanced debug functions
    debug_workflow.add_node("generate", enhanced_generate)    # Enhanced debug generation node
    debug_workflow.add_node("check_code", enhanced_code_check)  # Enhanced debug validation node
    debug_workflow.add_node("reflect", enhanced_reflect)      # Enhanced debug reflection node

    # Build graph connections
    debug_workflow.add_edge(START, "generate")      # Start with generation
    debug_workflow.add_edge("generate", "check_code")  # Check generated code

    # Add conditional routing based on code check result
    debug_workflow.add_conditional_edges(
        "check_code",
        enhanced_decide_to_finish,
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
    # Example usage
    state = {
        "messages": [],
        "iterations": 0,
        "error": "no",
        "generation": None
    }
    
    # Initialize the debug workflow
    debug_workflow = setup_debug_workflow()
    
    # Run the debug workflow
    debug_workflow.run(state)