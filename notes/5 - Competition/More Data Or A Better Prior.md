seems like its lackiong out of distrubtion datasets

Thought for 31s

There's a real version of that, and it's been tried. Michael Hodel's re-arc wrote procedural generators for all 400 ARC-1 training tasks so you could sample unlimited variants, and the BARC work generated a few hundred thousand synthetic ARC-like problems with LLMs. Both helped. Test-time training approaches on ARC-AGI-1 leaned on that data heavily, and numbers moved.

But notice what "out-of-distribution dataset" means once you've collected it. It's in-distribution now. You've moved the boundary, not removed it, and ARC-AGI-2 was built by looking at where the boundary sat and placing tasks past it. That's not a fixable oversight in the benchmark — it's the design. Any benchmark that more data solves is measuring accumulated skill, which is precisely what Chollet built ARC not to measure.

The decisive evidence is the human panel. Every ARC-AGI-2 task was solved by at least two ordinary people from San Diego in under two attempts, and not one of them had an ARC training set. They walked in with core knowledge from being alive and solved novel tasks from three examples. So data scarcity cannot be the explanation for the gap — the existence proof runs on zero.

And for your architecture specifically, more data cuts the wrong way. The problem I described wasn't coverage, it was ranking: too many programs consistent with three pairs. Growing the library adds predicates, which adds consistent hypotheses, which makes the ranking problem strictly worse. DreamCoder's compression step exists to fight that, and it's the part that doesn't scale to novel tasks. You'd be paying in search to buy expressiveness you already assumed you had.

Also worth saying plainly: "fix it with more training data" is the scaling move. It's the LLM camp's answer wearing symbolic clothes. If your bet is that structure beats data, then the shortage you're looking for isn't in the dataset — it's in the prior. That's a harder thing to build than a corpus, but it's also the only place where a symbolic entry has an edge nobody else is contesting.

excerpt_from_previous_claude_message.txt

1 line

TXT

this argues for more data not less

Thought for 25s

You're right and I overstated it. "Runs on zero" is false — a 30-year-old walked in with something like a billion seconds of embodied multimodal experience. That's a colossal pretraining budget. I should have said zero task-relevant data.

But the distinction that survives is the one that matters for your build. The human prior wasn't acquired from anything resembling ARC. It came from watching objects fall, tracking things behind occluders, inferring what other people wanted. Then it transferred to colored grids on a screen. Whereas re-arc and BARC are data that resembles the task — more grids, more transformations. If your argument is right, it argues specifically against the thing you originally proposed. It says: don't generate more ARC variants, go get physics and agency from somewhere else entirely. Cross-domain transfer, not distribution widening.

There's a developmental wrinkle too, and I'll flag its weakness myself since I criticized psych replication two messages ago: violation-of-expectation studies put object permanence at a few months, numerosity discrimination near birth, goal attribution around five or six months. That's a very small data budget for the specific priors Chollet built the benchmark on. Infant research has its own replication troubles, so I won't lean on it hard, but the timeline doesn't look like it needs a lifetime.

Where this actually lands, though, is friendlier to you than it sounds. Both camps now agree the answer is the prior, not the examples. The disagreement is only about how a prior gets in there. The neural bet is that you train it on the right distribution and hope objectness falls out. The symbolic bet is that objectness, contact, containment and agency are a short enough list to write down directly — skip the acquisition entirely, because you already know what's being acquired.

That's a defensible position, and it's the one your system is implicitly making. So the thing to be precise about is which of Spelke's systems you've actually encoded, and whether the ranking function prefers hypotheses that use them. Not more data. A stated prior, and a bias that pays it out