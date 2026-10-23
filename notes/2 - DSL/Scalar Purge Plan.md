# =======================================================
# STEP 1: PURGING THE SCALARS (Unarization Phase)
# =======================================================
# This document tracks all components to be removed from 
# the ARC DSL to eliminate the 'Scalar' type entirely.
# =======================================================

1. NODE PRIMITIVES (environment.hpp)
--------------------------------------
The following getters take Objects and return a Scalar. 
Now that all spatial/tiling logic relies on Node Variants, these are dead logic.
* GetColor
* GetX
* GetY
* Width
* Height
* CenterX
* CenterY
* Count

2. NODE TERMINALS (environment.hpp)
--------------------------------------
The following terminal provides raw Scalar numbers to the solver.
* Constant

3. LEGACY MATH NODES (environment.hpp - commented out but to be deleted)
--------------------------------------
* DeltaX
* DeltaY
* DirX
* DirY
* Add
* Subtract
* Multiply
* Sign

4. TYPE SYSTEM (environment.hpp)
--------------------------------------
* ast_type::Scalar (must be removed from the enum)
* Any references to ast_type::Scalar in get_return_type / get_expected_arg_type.

5. EXECUTOR LOGIC (dsl_executor.hpp)
--------------------------------------
* The entire 'eval_s' function block (the scalar evaluator).
* The 'exec_context::eval_s' signature.
* Any remaining dead switch-cases in 'execute' that handled Getters.
