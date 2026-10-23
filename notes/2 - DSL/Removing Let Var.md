Removing Let/Var in favor of the DAG Approach

In the ARC engine, we previously relied on Let(expr, body) and Var() AST nodes to store and reuse intermediate evaluation results. This relied on a thread_local vector<int> current_var buffer to stash the computed grid from expr, which Var() inside body would read.

Why it was flawed:
1. Single Variable Limit: Due to reliance on a single global buffer (current_var) and a depth counter (current_let_depth), it was difficult, slow, or impossible to correctly handle nested scopes or multiple variables. 
2. Brittle Constraints: The Beam Search logic required messy ad-hoc checks (e.g., is_root_let, enforcing has_var states, restricting refinement paths within Let nodes, or forcibly creating skeletons wrapped in Let(Input, Input)). 

The Solution: DAG (Directed Acyclic Graph) + Memoization
Instead of introducing programmatic scope and named variables, standard AST engines resolve this using structural equivalence and cache hits. We entirely dropped Let and Var. 

How computation reuse works now
If an algorithm needs to square a value (e.g., duplicate the result of Add(Input, 1) into two operands):

Old Way: 
Let(Add(Input, Constant(1)), Multiply(Var, Var))

New Way:
Multiply(Add(Input, Constant(1)), Add(Input, Constant(1)))

At first glance, the New Way appears to evaluate the Add operation twice. However, due to the engine's built-in structural hashing (environment::update_hash and get_cache_entry), identical subtrees effortlessly hit the cache.

Code:
// Inside eval_v
CacheEntry* c_entry = get_cache_entry(&ex, n);
if (c_entry->valid) {
    copy_n(c_entry->pixels, pixels, out.data());
    return;
}

When Multiply evaluates its left branch, it computes Add(Input, 1) and flags the hash + pixel grid as valid in the cache. When it evaluates its right branch, it encounters an identical structural hash, instantly returns the memoized pixels from the cache, and skips redundant calculations.

Conclusion
The Beam Search solver natively exploits this by inserting subtrees from its history_pool into multiple argument slots. This creates identical logical duplication to procedural variables effortlessly, with native C++ algorithms instead of scope-management logic.
