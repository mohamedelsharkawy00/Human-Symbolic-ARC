GOAL: Transitioning from Genetic Algorithm to Breadth-First Beam Search 
to support 'Zero-Bloat' systematic solver construction 
and enable effective Hindsight Learning.
DSL improvements will kill the need for search probably

- Add a refactor pass to the stohcastic solver
Brainstorming & Optimization Ideas

Shark Optimizations 2.0
- Consider making a scanning pass then an editing pass. each with their own trees maybe idk yet.
- DSL is missing memeory or lambda style memory/functionality and looping/bounded looping
- Hindsight: RLLLLLLLLLLLLLLL, and learning to combine mistakes/found behaviours 
  to fix problem at hand. 
  THE LEARNING TO COMBINE IS CRUCIAL.
- tell me first how to force it to generate only unique trees etc, 
  and also disssimalr simlar trees etc, brainstorming here, 
  also why does this solver not attempt to find simpler trees with time etc
also the decay should be from down to up right?

- Stop doing stochastic search and start doing beam exhaustive.
- You can combine methods from here, Iterative and beam for example.
- [IMPLEMENTED] Freeze good AST parts.
- [IMPLEMENTED] Gradient freeze from top to bottom.
- [IMPLEMENTED] Freeze all but the deepest nodes.
- [IMPLEMENTED] Freeze all but the terminals.
- [IMPLEMENTED] Freeze based on probability.
- [IMPLEMENTED] Freeze with decaying probability.
- Move to latent space? or other spaces where the solutions 
  maybe more closer/smoother to optimize between/hill climb them.

Dorr Optimizations
- The way to solve the problem of not converging after ading variable size grids, 
  is first to solve the constrained grid then test the solution 
  on the variable size grid. 
  If the solution is not correct, then we produce another solution 
  on the constrained then try again on the unconstrained.
- Can the AI ever think before attempting level?
- Start forcing the solver to view items like humans do: as objects and shapes.
    - Cluster either bottom-up, assuming each cell is an object, 
      or top-down, starting by assuming the grid is an object, 
      then smaller clusters as objects.
    - Reward/Bias the solver to try to find similarities.
- Splitting the solving trees to solve until a local optimum, 
  then solving the remaining error, is also promising.
- [PARTIALLY IMPLEMENTED] Reinforcement Learning on each puzzle on its own 
  (test-time training), with learning from mistakes and/or hindsight RL.
- For now: learning on the examples alone, so that bad trees are also scored 
  and better ones are ranked higher.
- Stratified by layers; either layers of clusters at different K/sizes, or by error.
- Research all DSL methods (SOTA and below) and all other non-LLM methods.
