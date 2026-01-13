import inspect, re, json
from litellm import completion, ModelResponse
type_map = {int: "integer", str: "string", float: "number", bool: "boolean"}
def mk_tool_def(fn):
    s = inspect.getsource(fn).split(')')[0]
    matches = re.findall(r'(\w+):.+#\s*(.+?)\s*$', s, flags=re.MULTILINE)
    params = inspect.signature(fn).parameters  # Changed from add_numbers to fn
    return {
        "type": "function", 
        "function": {
            "name": fn.__name__, 
            "description": fn.__doc__,  
            "parameters": {
                "type": "object",
                "properties": {v: {'type': type_map[fn.__annotations__[v]], 'description': d} for v,d in matches},
                "required": [p for p in params.keys() if params[p].default == inspect.Parameter.empty]
            }
        }

    }
def mk_msg(m, role='user'): return {"role":role, "content":m}
def mk_tool_res(m, tc_id): return mk_msg(m, role='tool') | {"tool_call_id": tc_id}
def ex_tool(tc):
    """Execute tool call"""
    fn = tc['function']
    res = str(globals()[fn['name']](**json.loads(fn['arguments'])))
    return mk_tool_res(res, tc['id'])
def _repr_markdown_(self):
    tool_info = ''
    if self.choices[0].finish_reason == 'tool_calls':
        tc = self.choices[0].message.tool_calls[0]
        fn = tc.function
        tool_info = f'\n\n\tFunction call: `{fn.name}(**{fn.arguments})`'
    return (self.choices[0].message.content or '') + tool_info

ModelResponse._repr_markdown_ = _repr_markdown_

class Chat:
    def __init__(self, model, tools=None, sp=None):
        self.model = model
        self.tools = tools and list(map(mk_tool_def, tools))
        self.msgs = [mk_msg(sp, 'system')] if sp else []
    
    def __call__(self, ct):
        self.msgs.append(mk_msg(ct))
        while True:
            res = completion(model=self.model, messages=self.msgs, tools=self.tools)
            self.msgs.append(res.choices[0].message)
            if not (tcs := res['choices'][0]['message']['tool_calls']): break
            for tc in tcs: self.msgs.append(ex_tool(tc))
        return res
def add_numbers(
    a: int,  # First number to add
    b: int   # Second number to add  
) -> int:
    "Add two numbers together"
    return a + b
