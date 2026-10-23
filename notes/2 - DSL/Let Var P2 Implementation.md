REMOVED LET/VAR IMPLEMENTATION (REFERENCE)

1. Node Types & Macros

#define NODE_PRIMITIVES \
    ... \
    X(Let, 2, "Let", 5) /* (val, body) */ \
    ...

#define NODE_TERMINALS \
    ... \
    X(Var, 0, "Var", 5) \
    ...

2. Execution Logic (eval_v)

// In execute(...) function
static thread_local vector<vector<int> *> var_stack;
var_stack.clear();

auto eval_v = [&](auto &&self, const node *n, vector<int> &out) -> void {
    ...
    if (arity == 0) {
        if (n->type == node_type::Var) {
            if (!var_stack.empty()) {
                const auto &val = *var_stack.back();
                if (val.size() == (size_t)pixels)
                    std::copy(val.begin(), val.end(), out.begin());
                else
                    fill(out.begin(), out.begin() + pixels, 0);
            }
            return;
        }
    ...
    if (n->type == node_type::Let) {
        auto &rhs = acquire();
        self(self, n->args[0], rhs);
        var_stack.push_back(&rhs);
        self(self, n->args[1], out);
        var_stack.pop_back();
        release();
        release();
        return;
    }
    ...
};

3. Pruning & Simplification

static inline bool uses_var(const node *n) {
    if (!n) return false;
    if (n->type == node_type::Var) return true;
    int arity = type_arity[to_underlying(n->type)];
    for (int i = 0; i < arity; ++i)
        if (uses_var(n->args[i])) return true;
    return false;
}

// is_reducible rules
if (type == node_type::Let) {
    if (!uses_var(args[1])) return true; // Body doesn't use Var
    if (args[0]->type == node_type::Constant || args[0]->type == node_type::Var)
        return true; // Don't bind constants or variables
}

// simplify rules
if (n->type == node_type::Let) {
    if (!uses_var(n->args[1])) {
        node *r = copy(n->args[1]);
        remove(n);
        return r;
    }
    if (n->args[1]->type == node_type::Var) {
        node *r = copy(n->args[0]);
        remove(n);
        return r;
    }
}

4. Node Weighting

static inline int count_nodes(const node *n) {
    if (!n) return 0;
    // Treating Let as 5 nodes gives it a higher barrier to entry
    int c = (n->type == node_type::Let) ? 5 : 1;
    for (int i = 0; i < MAX_ARITY; ++i)
        c += count_nodes(n->args[i]);
    return c;
}
