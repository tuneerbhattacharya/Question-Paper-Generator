"""A tiny stand-in for LangGraph used ONLY when langgraph isn't installed (tests run offline).

In CI, requirements install the real langgraph and the tests run against it instead.
Supports what qpg.graph uses: add_node, add_edge, add_conditional_edges, compile, invoke,
stream(stream_mode="values"), Annotated reducers and recursion_limit.
"""
import typing

START, END = "__start__", "__end__"


class GraphRecursionError(RuntimeError):
    pass


class StateGraph:
    def __init__(self, schema):
        hints = typing.get_type_hints(schema, include_extras=True)
        self.reducers = {k: v.__metadata__[0] for k, v in hints.items()
                         if hasattr(v, "__metadata__")}
        self.nodes, self.edges, self.cond = {}, {}, {}

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, a, b):
        self.edges[a] = b

    def add_conditional_edges(self, a, router, mapping):
        self.cond[a] = (router, mapping)

    def compile(self):
        return self

    def _next(self, node, state):
        if node in self.cond:
            router, mapping = self.cond[node]
            return mapping[router(state)]
        return self.edges[node]

    def stream(self, state, config=None, stream_mode="values"):
        limit = (config or {}).get("recursion_limit", 25)
        state = dict(state)
        node, steps = self._next(START, state), 0
        while node != END:
            steps += 1
            if steps > limit:
                raise GraphRecursionError(f"recursion limit {limit} reached")
            for key, value in self.nodes[node](state).items():
                if key in self.reducers and key in state:
                    state[key] = self.reducers[key](state[key], value)
                else:
                    state[key] = value
            yield dict(state)
            node = self._next(node, state)

    def invoke(self, state, config=None):
        final = state
        for final in self.stream(state, config):
            pass
        return final
