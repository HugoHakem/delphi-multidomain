import ast, inspect, types, textwrap
from easydict import EasyDict

history = {}

def hook_assign(name, value):
    if name not in history:
        history[name] = []
    history[name].append(value.clone() if hasattr(value, "clone") else value)
    return value

class AssignHooker(ast.NodeTransformer):

    def visit_Assign(self, node):
        self.generic_visit(node)
        new_targets = []
        for t in node.targets:
            if isinstance(t, ast.Name):
                # x = expr  →  x = hook_assign("x", expr)
                node.value = ast.Call(
                    func=ast.Name(id="hook_assign", ctx=ast.Load()),
                    args=[ast.Constant(t.id), node.value],
                    keywords=[]
                )
                new_targets.append(t)
        node.targets = new_targets
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            # Reescribimos: x += expr
            # a: expr
            #   ↓
            # x = hook_assign("x", x <op> expr)
            new_value = ast.BinOp(
                left=ast.Name(id=node.target.id, ctx=ast.Load()),
                op=node.op,
                right=node.value,
            )
            return ast.Assign(
                targets=[node.target],
                value=ast.Call(
                    func=ast.Name(id="hook_assign", ctx=ast.Load()),
                    args=[ast.Constant(node.target.id), new_value],
                    keywords=[],
                )
            )
        return node


def instrument(func):
    src = inspect.getsource(func)
    src = textwrap.dedent(src)
    
    tree = AssignHooker().visit(tree:=ast.parse(src))
    ast.fix_missing_locations(tree)

    # ns = {}
    exec(
        compile(tree, "<ast>", "exec"),
        {**func.__globals__, "hook_assign": hook_assign, "history": history},
        ( ns:={} )
    )
    new_func = ns[func.__name__]
    ######################################################### 
    if hasattr(func, "__self__"):
        new_func = types.MethodType(new_func, func.__self__)
    return lambda *x: (new_func(*x), EasyDict(history))
