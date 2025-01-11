"""

AutoModule 自动化代码 打包

"""








"""           auto_module.py           """


import torch
import torch.nn as nn

import inspect


class AutoModule(nn.Module):
    def __init__(self, args_local):
        super().__init__()

        self.auto_states = {}

        self.args_local, self.args_init = dict(args_local), {}
        self.self_class = self.args_local["__class__"]
        self.kwargs_pass = args_local.get('kwargs', None)
        for name in args_local.keys():
            if name not in ['self', '__class__', 'kwargs']:
                self.args_init[name] = args_local[name]
                setattr(self.args_local['self'], name, args_local[name])

        # if 'kwargs' not in args_local.keys():
        #     print(f'AutoModule Warning: In {self.self_class}, '
        #           f'there no **kwargs that AutoModule use it to pass param')

        # todo add a instantiate check, if a AutoModule is not instantiate by self.instantiate, throw a warning

        self.dtype = None
        self.device = None

    def instantiate(self, module, **kwargs):
        args_module = inspect.signature(module).parameters

        names_empty = []
        names_default = {}
        for name_param, default_param in map(lambda m: (m.name, m.default), args_module.values()):
            if name_param != 'kwargs':
                if default_param is inspect.Parameter.empty:
                    names_empty.append(name_param)
                else:
                    names_default[name_param] = default_param

        param = dict(self.kwargs_pass) if self.kwargs_pass is not None else {}
        for name, value in self.args_init.items():
            if not callable(value):
                param[name] = value

        param = {name: param[name] for name in param.keys() - names_default.keys()}

        param.update(kwargs)

        if 'kwargs' not in args_module:
            param_fit = {name: param[name] for name in param.keys() & names_empty}
            return module(**param_fit)
        else:
            return module(**param)

    def to(self, *args, **kwargs):

        try:
            dtype = kwargs.get('dtype', None) or next(filter(lambda arg: isinstance(arg, torch.dtype), args))
        except StopIteration:
            dtype = None

        try:
            device = kwargs.get('device', None) or next(filter(lambda arg: isinstance(arg, torch.device), args))
        except StopIteration:
            device = None

        self.dtype = dtype or self.dtype
        self.device = device or self.device

        for module in self.modules():
            if isinstance(module, AutoModule):

                module.dtype = self.dtype
                module.device = self.device

                for auto in module.auto_states.values():
                    if isinstance(auto.state, torch.Tensor):
                        dtype = auto.state.dtype
                        dtype = self.dtype if dtype.is_floating_point or dtype.is_complex else None
                        auto.state = auto.state.to(device=self.device, dtype=dtype)

        return super().to(*args, **kwargs)

    class AutoState:
        def __init__(self):
            self.inited = False
            self.state = None

        def if_not(self, init):
            if not self.inited:
                self.update(init())
            return self.state

        def __call__(self, init):
            return self.if_not(init)

        def update(self, value):
            self.inited, self.state = True, value

        def reset(self, inited=False, state=None):
            self.inited, self.state = inited, state

    def declare_auto_state(self, name):
        assert name not in self.auto_states.keys()

        auto = self.AutoState()
        self.auto_states[name] = auto
        return auto

    def get_declared_auto_state(self, name):
        return self.auto_states[name]

    def reset_all_auto_state(self, recursive=True):
        if recursive:
            for module in self.modules():
                if isinstance(module, AutoModule):
                    module.reset_all_auto_state(recursive=False)
        else:
            for auto in self.auto_states.values():
                auto.reset()

    def get_all_auto_state(self, recursive=True, detach_tensor=True):
        if recursive:
            state = dict(self.named_modules())
            for name, module in state.items():
                state[name] = module.get_all_auto_state(recursive=False) if isinstance(module, AutoModule) else {}
            return state
        else:
            state = dict(self.auto_states)
            for name, auto in state.items():
                new = self.AutoState()
                new.inited, new.state = auto.inited, auto.state
                if detach_tensor:
                    if isinstance(new.state, torch.Tensor):
                        new.state = new.state.clone().detach().cpu()
                state[name] = new
            return state

    def set_all_auto_state(self, state, recursive=True, attach_tensor=True):
        if recursive:
            for name, module in dict(self.named_modules()).items():
                if isinstance(module, AutoModule):
                    module.set_all_auto_state(state[name], recursive=False)
        else:
            for name, new in state.items():
                auto = self.auto_states[name]
                auto.inited, auto.state = new.inited, new.state
                if attach_tensor:
                    if isinstance(auto.state, torch.Tensor):
                        auto.state = auto.state.to(device=self.device)

    def find_state(self, name_state):
        state = {}
        for name_module, module in dict(self.named_modules()).items():
            if isinstance(module, AutoModule):
                if name_state in module.auto_states:
                    state[name_module] = module.auto_states[name_state]
        return state

    def temp_state(self):
        class temp_area:
            def __init__(self, model):
                self.model = model
                self.state = None

            def __enter__(self):
                self.state = self.model.get_all_auto_state()
                self.model.reset_all_auto_state()

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.model.set_all_auto_state(self.state)

        return temp_area(self)

    def with_temp_state(self, func):
        with self.temp_state():
            func()

    @staticmethod
    def phase_flow(flow):
        name_in, name_out = flow.replace(' ', '').split('->')
        name_in, name_out = name_in.split(','), name_out.split(',')
        assert all(map(lambda x: x.isidentifier(), name_in + name_out)), \
            f'Not a valid identifier; In flow instruction: {name_in} -> {name_out}.'
        assert len(set(name_out)) == len(name_out), \
            f'Duplicated identifier; In flow instruction returning: -> {name_out}.'
        return name_in, name_out

    @staticmethod
    def phase_namespace_returning(module):
        if hasattr(module, 'namespace'):
            name_in = module.namespace
        else:
            name_in = tuple(inspect.signature(module.forward).parameters)

        if hasattr(module, 'returning'):
            name_out = module.returning
        else:
            line_return = inspect.getsourcelines(module.forward)[0][-1]
            name_out = tuple(line_return.strip().lstrip('return').replace(' ', '').split(','))

        assert all(map(lambda x: x.isidentifier(), name_in + name_out)), \
            f'Not a valid identifier; In module namespace or returning: {name_in} -> {name_out}.'
        assert len(set(name_out)) == len(name_out), \
            f'Duplicated identifier; In module returning: -> {name_out}.'

        return name_in, name_out

    def __repr__(self):
        rep = super().__repr__()
        idx = rep.find('\n')

        namespace, returning = self.phase_namespace_returning(self.args_local['self'])
        namespace, returning = ','.join(namespace), ','.join(returning)

        args_repr = []
        for name, value in self.args_init.items():
            if not callable(value) and name not in ('do', 'module_list'):
                args_repr.append(f' {name}={value}')
        args_repr = ','.join(args_repr)

        if args_repr != '':
            args_repr = ''.join(['  &  ', args_repr])

        return ''.join((rep[:idx], namespace, '->', returning, args_repr, rep[idx:]))


if __name__ == '__main__':
    pass
    # test1()







"""           auto_comp.py           """


import inspect
import torch
import torch.nn as nn

# from auto import AutoModule


class AutoSequential(AutoModule, nn.ModuleList):
    def __init__(self, module_list, do, **kwargs):
        super().__init__(locals())

        for index, module in enumerate(module_list):
            self.append(self.instantiate(module, idx_sequent=kwargs.get('idx_sequent', ()) + (index,)))

        self.list_name_in, self.list_name_out = [], []
        self.namespace, self.returning = self.phase_flow(do)
        assert len(self.namespace) > 0 and len(self.returning) > 0

        for module in self:
            name_in, name_out = self.phase_namespace_returning(module)
            self.list_name_in.append(name_in)
            self.list_name_out.append(name_out)

    def forward(self, *args):
        assert len(args) == len(self.namespace)
        kwargs = dict(zip(self.namespace, args))

        for index, module in enumerate(self):
            inputs = tuple(map(lambda x: kwargs[x], self.list_name_in[index]))
            kwargs.update(dict(zip(
                self.list_name_out[index],
                type(outputs := module(*inputs)) in (tuple, list) and outputs or (outputs,)
            )))

        return tuple(map(lambda x: kwargs[x], self.returning)) if len(self.returning) > 1 \
            else kwargs[self.returning[0]]


class AutoLayer(AutoModule, nn.ModuleList):
    def __init__(self, module, num_layer, do=None, **kwargs):
        super().__init__(locals())

        for index in range(num_layer):
            self.append(self.instantiate(module, idx_layer=kwargs.get('idx_layer', ()) + (index,)))

        self.namespace, self.returning = do and self.phase_flow(do) or self.phase_namespace_returning(self[0])
        assert len(self.namespace) == len(self.returning)

    def forward(self, *args):
        assert len(args) == len(self.namespace)
        kwargs = dict(zip(self.namespace, args))

        for module in self:
            inputs = tuple(map(lambda x: kwargs[x], self.namespace))
            kwargs.update(dict(zip(
                self.returning,
                type(outputs := module(*inputs)) in (tuple, list) and outputs or (outputs,)
            )))

        return tuple(map(lambda x: kwargs[x], self.returning)) if len(self.returning) > 1 \
            else kwargs[self.returning[0]]


class AutoDiverse(AutoModule, nn.ModuleList):
    def __init__(self, module, do, **kwargs):
        super().__init__(locals())

        self.namespace, self.returning = self.phase_flow(do)
        assert len(self.returning) > 0

        for index in range(len(self.returning)):
            self.append(self.instantiate(module, idx_diverse=kwargs.get('idx_diverse', ()) + (index,)))

        if len(self.namespace) == 0:
            if hasattr(self[0], 'namespace'):
                self.namespace = self[0].namespace
            else:
                self.namespace = tuple(inspect.signature(self[0].forward).parameters)

    def forward(self, *args):
        assert len(args) == len(self.namespace)
        kwargs = dict(zip(self.namespace, args))

        outputs = []
        for module in self:
            inputs = tuple(map(lambda x: kwargs[x], self.namespace))
            outputs.append(module(*inputs))

        return tuple(outputs) if len(outputs) > 1 else outputs[0]


class AutoSum(AutoModule):
    def __init__(self, do, **kwargs):
        super().__init__(locals())
        self.namespace, self.returning = self.phase_flow(do)

    def forward(self, *args):
        assert len(args) == len(self.namespace)
        x = torch.cat(tuple(map(lambda t: t.unsqueeze(dim=0), args)), dim=0).sum(dim=0)
        x = x.tile(len(self.returning), *(1,) * (len(x.shape) - 1)).chunk(len(self.returning), dim=0)
        return x if len(self.returning) > 1 else x[0]


class AutoPack(AutoModule):
    def __init__(self, module, do, **kwargs):
        super().__init__(locals())
        self.module = self.instantiate(module)
        self.namespace, self.returning = self.phase_flow(do)

    def forward(self, *args):
        assert len(args) == len(self.namespace)
        outputs = type(outputs := self.module(*args)) in (tuple, list) and outputs or (outputs, )
        assert len(outputs) == len(self.returning)
        return outputs if len(outputs) > 1 else outputs[0]


class AutoIdentity(AutoModule):
    def __init__(self, do='args->args'):
        super().__init__(locals())
        self.namespace, self.returning = self.phase_flow(do)
        assert len(self.namespace) == len(self.returning)

    def forward(self, *args):
        if len(self.namespace) > 1:
            assert len(args) == len(self.namespace)
        return args if len(args) > 1 else args[0]


class AutoDo(AutoModule):
    def __init__(self, do):
        super().__init__(locals())
        pass


class AutoExec(AutoModule):
    def __init__(self, do, code):
        super().__init__(locals())
        pass


if __name__ == '__main__':
    pass
    # comp_test()



"""           auto_kernel.py            """



import inspect
import re

import torch
from torch.utils.cpp_extension import load_inline

# from auto import AutoModule


class AutoKernel(AutoModule):
    def __init__(self, args_local, forward, backward):
        super().__init__(args_local)
        self.kernel = None
        self.namespace, self.returning = None, None
        self.kernel_forward, self.kernel_backward = None, None

        assert inspect.getsourcelines(self.self_class.forward) == inspect.getsourcelines(self.forward), \
            f'AutoKernel: Func forward() should not be rewrite, use other names. '

        self.link(forward, backward)

    def forward(self, *args):
        return self.kernel.apply(self.kernel_forward, self.kernel_backward, *args)

    @staticmethod
    def __forward(ctx, forward, backward, *args):
        with torch.no_grad():
            args = map(lambda x: x.contiguous() if isinstance(x, torch.Tensor) else x, args)
            ctx.auto_cuda_kernel_pass_backward_func__ = backward
            return forward(ctx, *args)

    @staticmethod
    def __backward(ctx, *grad):
        with torch.no_grad():
            grad = map(lambda x: x.contiguous() if isinstance(x, torch.Tensor) else x, grad)
            return None, None, *ctx.auto_cuda_kernel_pass_backward_func__(ctx, *grad)

    def link(self, forward, backward):
        """
        :param forward: func (self, ctx, *args)
        :param backward: func (self, ctx, *grad)
        """

        # identify = '@'.join([str(forward), str(backward)])

        self.kernel_forward = forward
        self.kernel_backward = backward

        self.namespace = tuple(inspect.signature(forward).parameters)[1:]

        line_return = inspect.getsourcelines(forward)[0][-1]
        self.returning = tuple(line_return.strip().lstrip('return').replace(' ', '').split(','))

        kernel = type('', (torch.autograd.Function,), {
            'forward': self.__forward,
            'backward': self.__backward,
        })

        self.kernel = kernel()

    @staticmethod
    def grad_check(kernel, inputs, dtype=torch.float64, device=torch.device('cuda')):
        """compare analysis grads and numerical grad, call like this : grad_check(kernel, [q, k, v])"""

        inputs = list(map(lambda arg: arg.clone().detach(), inputs))
        inputs = list(map(lambda arg: arg.to(dtype), inputs))
        inputs = list(map(lambda arg: arg.to(device), inputs))
        inputs = list(map(lambda arg: arg.requires_grad_(), inputs))

        print('=============================')
        print('Comparing analytical grads and numerical grads:')
        check = torch.autograd.gradcheck(kernel, inputs,
                                         # eps=1e-6, atol=1e-5, rtol=1e-3,
                                         raise_exception=True)
        print(f'check -> {check}')
        print('=============================')

    @staticmethod
    def diff_check(kernel, naive, inputs, name_inputs, name_outputs, dtype=torch.float64, device=torch.device('cuda')):
        """compare outputs and grads of 2 forms, call like this : diff_check(form1, form2, [q, k, v], 'qkv', 'y')"""

        inputs = list(map(lambda arg: arg.clone().detach(), inputs))

        def diff(a, b):
            return (a - b).abs().mean().item(), (a - b).abs().max().item()

        def loss(x):
            return ((x * x) - torch.tanh(x)).sum()

        def test(func, tensors, idx_output):
            tensors = list(map(lambda arg: arg.clone().detach(), tensors))
            tensors = list(map(lambda arg: arg.to(dtype), tensors))
            tensors = list(map(lambda arg: arg.to(device), tensors))
            tensors = list(map(lambda arg: arg.requires_grad_(), tensors))
            output = func(*tensors)[idx_output]
            loss(output).backward()
            grads = list(map(
                lambda arg: arg.grad.clone() if arg.grad is not None else torch.zeros(1).to(device),
                tensors
            ))

            return output, grads

        print('=============================')
        print('abs(a-b)......(mean, max)')
        print('-----------------------------')
        for index, name_o in enumerate(name_outputs):
            output_kernel, grads_kernel = test(kernel, inputs, index)
            output_naive, grads_naive = test(naive, inputs, index)

            print(name_o + '    ', diff(output_kernel, output_naive))
            for grad_k, grad_n, name_i in zip(grads_kernel, grads_naive, name_inputs):
                print(name_o + '/d ' + name_i, diff(grad_k, grad_n))
            print('-----------------------------')
        print('=============================')


class AutoCudaKernel:
    _cuda = {}
    _count = 0

    _type_name = {
        torch.bfloat16: 'at::BFloat16',
        torch.float32: 'float',
        torch.float64: 'double',
        # torch.bool: 'bool',
        # torch.long: 'int64_t'
    }

    @staticmethod
    def to_ctype(torch_type):
        return AutoCudaKernel._type_name[torch_type]

    @staticmethod
    def link(url, forward, backward, flags=None, verbose=True):

        flags = flags or []

        with open(url, 'r') as cuda:
            code_cuda = cuda.read()

        identify = '@'.join([code_cuda, forward, backward] + flags)
        if identify not in AutoCudaKernel._cuda:
            name = url[url.rfind('/') + 1:].rstrip('.cu') + f'_{AutoCudaKernel._count}'
            AutoCudaKernel._count += 1
            code_cpp, code_cuda = AutoCudaKernel.generate_cpp(name, code_cuda, forward, backward)
            AutoCudaKernel._cuda[identify] = AutoCudaKernel.compile_cuda(name,
                                                                         code_cpp=code_cpp,
                                                                         code_cuda=code_cuda,
                                                                         c_flags=flags,
                                                                         c_verbose=verbose)

        return AutoCudaKernel._cuda[identify]

    @staticmethod
    def compile_cuda(name, code_cpp, code_cuda, c_flags, c_verbose):

        cuda = load_inline(name=name,
                           cpp_sources=code_cpp,
                           cuda_sources=code_cuda,
                           verbose=c_verbose,
                           extra_cflags=c_flags,
                           extra_cuda_cflags=['-res-usage',
                                              '--use_fast_math',
                                              '-O3',
                                              '-Xptxas -O3',
                                              '--extra-device-vectorization',
                                              ] + c_flags)

        return cuda

    @staticmethod
    def generate_cpp(name, code_cuda, name_forward, name_backward):

        code_cpp = ['#include <torch/extension.h>\n'
                    '#include "ATen/ATen.h"\n'
                    'AUTO_CUDA_KERNEL_PRE_DEFINE\n'
                    '// Auto Generated Code\n']

        def phase_args(code, link):
            patten_func = r'void\s+' + link + r'\s*(?P<args>\([\w\s,*]+\))'
            define_func = list(re.finditer(patten_func, code))
            assert len(define_func) == 1
            patten_args = r'(?P<type>\w+[\s*]+)(?P<name>\w+)'
            define_args = list(re.finditer(patten_args, define_func[0]['args']))

            return define_args

        def add_func(name_func, args_list):

            code = ['void ' + name_func + '(']
            for index, args in enumerate(args_list):
                code.append(args['type'] + args['name'])
                if index < len(args_list) - 1:
                    code.append(',')
            code.append(');\n')

            code.append('void ' + name_func + '_' + '(')
            for index, args in enumerate(args_list):
                if '*' in args['type']:
                    code.append('torch::Tensor & ')
                elif 'int' in args['type']:
                    code.append('int64_t ')
                else:
                    assert False, 'Unknown Type'
                code.append(args['name'])
                if index < len(args_list) - 1:
                    code.append(',')
            code.append('){\n' + name_func + '(')

            for index, args in enumerate(args_list):
                code.append(args['name'])
                if '*' in args['type']:
                    type_data = args['type'].replace('*', '')
                    code.append(f'.data_ptr<{type_data}>()')
                if index < len(args_list) - 1:
                    code.append(',')
            code.append(');}\n')

            return code

        args_forward = phase_args(code_cuda, name_forward)
        code_cpp += add_func(name_forward, args_forward)

        args_backward = phase_args(code_cuda, name_backward)
        code_cpp += add_func(name_backward, args_backward)

        # todo 支持多个函数 而不是固定的前向反向

        code_cpp.append(f'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) '
                        '{'
                        f'm.def("forward", &{name_forward}_, "{name} forward");'
                        f'm.def("backward", &{name_backward}_, "{name} backward");'
                        '}\n')

        code_cpp.append(f'TORCH_LIBRARY({name}, m) '
                        '{'
                        f'm.def("forward", {name_forward}_);'
                        f'm.def("backward", {name_backward}_);'
                        '}\n')

        code_cpp = ''.join(code_cpp)

        return code_cpp, code_cuda





"""              auto_logging.py               """



import os
import json
import datetime

import torch
from tqdm.auto import tqdm


class AutoLog:
    def __init__(self, url, force_reload=False):
        self.url = url

        self.dir_path, self.filename = os.path.split(url)
        assert self.filename != ''

        if os.path.exists(url):
            assert force_reload, 'AutoLog: To loc a exited log, set the force_reload to True.'
            assert len(list(self.list_log('__creat_log_file__'))) == 1
            self.update_log('__loc_log_file__', {'dir': self.dir_path, 'name': self.filename})
            print(f'AutoLog: Loc File {url}.')
        else:
            if not os.path.exists(self.dir_path):
                os.makedirs(self.dir_path, exist_ok=True)

            self.update_log('__creat_log_file__', {'dir': self.dir_path, 'name': self.filename})
            print(f'AutoLog: Created Log {url}')

    @classmethod
    def readonly(cls, url):
        class AutoLogReadOnly:
            def __init__(self, log_file):
                assert os.path.isfile(log_file)
                with open(log_file, 'r') as log:
                    self.log = tuple(map(lambda line: json.loads(line), log.readlines()))

            def list_log(self, tag, meta_info=False):
                lines = list(filter(lambda line: line['tag'] == tag, self.log))
                # lines = sorted(lines, key=lambda line: line['time'])
                info = map(lambda line: line['info'], lines)
                if meta_info:
                    meta = map(lambda line: {
                        'time': line['time'], 'logger': line['logger'], 'version': line['version']
                    }, lines)
                    return list(info), list(meta)
                else:
                    return list(info)
        return AutoLogReadOnly(url)

    def update_log(self, tag, info):
        time = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
        json_line = json.dumps({'tag': tag, 'time': time, 'info': info, 'logger': repr(self), 'version': 0})
        with open(self.url, 'a') as log:
            log.write(''.join([json_line, '\n']))

    def snapshot_script(self, script_file):
        with open(script_file, 'r') as file:
            script = file.read()
            self.update_log('__snapshot_script__', {'script': script})

    def save_model(self, model, incremental=True, info=None, name=None):
        if name is None:
            if not incremental:
                name, _ = self.list_saved_model()
                if len(name) == 0:
                    incremental = True
                else:
                    name = name[-1]
            if incremental:
                time = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
                name = ''.join([time, '-', hex(id(model)), '.pth'])

        model_url = os.path.join(self.dir_path, name)
        torch.save(model.state_dict(), model_url)
        self.update_log('__save_model__', {'name': name, 'incremental': incremental, 'info': info})
        print(f'AutoLog: Saved model {name}, incremental={incremental}, info={info}')
        return name

    def list_saved_model(self):
        model_saved = {}
        for info, meta in zip(*self.list_log('__save_model__', meta_info=True)):
            model_saved[info['name']] = meta['time']
        return zip(*sorted(model_saved.items(), key=lambda i: i[1])) if len(model_saved) > 0 else ((), ())

    def load_model(self, model, name, info, strict=True):
        model.load_state_dict(torch.load(os.path.join(self.dir_path, name)), strict=strict)
        self.update_log('__load_model__', {'name': name, 'id_model': hex(id(model)), 'info': info})
        print(f'AutoLog: Load model {name}, info={info}')

    @staticmethod
    def loc_info(loc, names, func=None):
        return {name: func and func(loc[name]) or loc[name] for name in names}

    def load_newest_saved_model(self, model):
        pass

    def list_log(self, tag, meta_info=False):
        with open(self.url, 'r') as log:
            lines = log.readlines()
            lines = map(lambda line: json.loads(line), lines)
            lines = filter(lambda line: line['tag'] == tag, lines)
            # lines = sorted(lines, key=lambda line: line['time'])
            lines = list(lines)

            info = map(lambda line: line['info'], lines)
            if meta_info:
                meta = map(lambda line: {
                    'time': line['time'], 'logger': line['logger'], 'version': line['version']
                }, lines)
                return list(info), list(meta)
            else:
                return list(info)


class AutoProcessBar:
    def __init__(self, max_length=100):
        self.process = None
        self.process_bar = None
        self.process_max_length = max_length

    def init_process_bar(self, max_length=None):
        self.process_max_length = max_length or self.process_max_length
        self.close_process_bar()

    def set_length(self, max_length):
        self.process_max_length = max_length

    def step_process(self, description=None):
        try:
            next(self.process)
        except (TypeError, StopIteration):
            self.process_bar = tqdm(range(self.process_max_length),
                                    total=self.process_max_length,
                                    ncols=128,
                                    bar_format='{l_bar}{bar:8}{r_bar}')
            self.process = iter(self.process_bar)
            next(self.process)

        if description is not None:
            self.process_bar.set_description_str(description)

    def close_process_bar(self):
        if self.process_bar is not None:
            self.process_bar.close()
            self.process = None
            self.process_bar = None


class AutoGroupLog:
    pass


if __name__ == '__main__':
    pass
    # test()


"""           auto_log_analysis.py        """


# import numpy as np
# import matplotlib.pyplot as plt

class AutoAnalysis:
    pass


"""           auto_multi_case.py        """


import inspect

import torch
import torch.nn as nn

import multiprocessing


class AutoMultiCase:
    pass


