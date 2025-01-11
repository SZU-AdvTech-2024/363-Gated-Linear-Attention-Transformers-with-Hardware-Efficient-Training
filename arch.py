


from auto import *




# from auto import *

from functools import partial


class TokenPredictor(AutoModule):
    def __init__(self, module, embedding, projector, normalizer=AutoIdentity, pos_embedding=AutoIdentity,
                 tying_weight=False, embedding_init_std=-1, **kwargs):
        super().__init__(locals())

        self.embedding = self.instantiate(embedding)
        self.pos_embedding = self.instantiate(pos_embedding)
        self.module = self.instantiate(module)
        self.normalizer = self.instantiate(normalizer)
        self.projector = self.instantiate(projector)

        if embedding_init_std >= 0:
            nn.init.normal_(self.embedding.weight, std=embedding_init_std)

        if tying_weight:
            self.projector.weight = self.embedding.weight

    def forward(self, x):
        """
        :param x: A tensor (siz_batch, num_token)
        :return x: A tensor (siz_batch, num_token, num_unique)
        """
        for module in (self.embedding, self.pos_embedding, self.module, self.normalizer, self.projector):
            x = module(x)
        return x

    def step(self, x, num_step=1, recorder=None):
        """
        :param x: A tensor (siz_batch, num_token)
        :param num_step: Predict length
        :param recorder: A function that use to return data step by step
        :return x: A tensor (siz_batch, num_token, num_unique)
        """
        x = self.forward(x)

        out = [torch.argmax(x[:, -1:], dim=-1)]
        if recorder is not None:
            recorder(out[-1])

        for i in range(num_step):
            out.append(torch.argmax(self.forward(out[-1]), dim=-1))
            if recorder is not None:
                recorder(out[-1])

        return torch.cat(out, dim=1)


class TimeChannelMixingBlock(AutoModule):
    def __init__(self, time_mixing, channel_mixing, normalizer, type_ln2res='pre', **kwargs):
        super().__init__(locals())

        self.block = self.instantiate({
            'pre':
                partial(AutoSequential, do='x->x', module_list=[
                    partial(AutoSum, do='x -> res'),
                    normalizer, time_mixing,
                    partial(AutoSum, do='x, res -> x, res'),
                    normalizer, channel_mixing,
                    partial(AutoSum, do='x, res -> x'),
                ]),
            'post':
                partial(AutoSequential, do='x->x', module_list=[
                    normalizer, partial(AutoSum, do='x -> res'),
                    time_mixing, partial(AutoSum, do='x, res -> x'),
                    normalizer, partial(AutoSum, do='x -> res'),
                    channel_mixing, partial(AutoSum, do='x, res -> x'),
                ])
        }[type_ln2res])

    def forward(self, x):
        x = self.block(x)
        return x


class TokenShift(AutoModule):
    def __init__(self, dim_feature, do='x->x', siz_kernel=2):
        super().__init__(locals())

        self.namespace, self.returning = self.phase_flow(do)
        assert len(self.namespace) == 1

        self.x_cross_chunk = self.declare_auto_state('x_cross')

        self.shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.mixing_list = nn.ParameterList([
            nn.Parameter(nn.init.uniform_(torch.zeros(dim_feature), a=0.5, b=1.0)) for _ in range(len(self.returning))
        ])

    def forward(self, x):
        """:param x: A tensor (siz_batch, num_token, dim_feature) """

        siz_batch, _, _ = x.shape

        last = self.x_cross_chunk(lambda: torch.zeros(siz_batch, self.dim_feature).to(x))

        x_shifted = self.shift(x)
        x_shifted[:, 0, :] = last

        self.x_cross_chunk.update(x[:, -1, :].clone().detach())

        token_shift = []
        for mixing in self.mixing_list:
            token_shift.append(x * mixing + x_shifted * (1 - mixing))

        return token_shift


class LowRankLinear(AutoModule):
    def __init__(self, dim_feature, low_rank, do='x->x', dim_output=None, bias_in=False, bias_out=False):
        super().__init__(locals())

        self.namespace, self.returning = self.phase_flow(do)
        assert len(self.namespace) == len(self.returning)

        self.dim_output = dim_output or dim_feature

        self.linear_in_list = nn.ModuleList(
            [nn.Linear(dim_feature, low_rank, bias=bias_in) for _ in range(len(self.namespace))]
        )

        self.linear_out_list = nn.ModuleList(
            [nn.Linear(low_rank, self.dim_output, bias=bias_out) for _ in range(len(self.namespace))]
        )

        # init weight
        for linear in self.linear_in_list + self.linear_out_list:
            nn.init.normal_(linear.weight, std=0.002)
            if linear.bias is not None:
                nn.init.constant_(linear.bias, 0)

    def forward(self, *args):
        """:param args: A Tuple of tensor (... , dim_feature) """

        output = []
        for index, (linear_in, linear_out) in enumerate(zip(self.linear_in_list, self.linear_out_list)):
            x = args[index]
            x = linear_out(torch.tanh(linear_in(x)))
            output.append(x)
        return output[0] if len(output) == 1 else tuple(output)


class GroupNorm(AutoModule):
    def __init__(self, dim_feature, num_head, affine=True, do='x->x'):
        super().__init__(locals())

        self.namespace, self.returning = self.phase_flow(do)
        assert len(self.namespace) == len(self.returning) == 1

        self.norm = nn.GroupNorm(num_groups=num_head, num_channels=dim_feature, affine=affine)

    def forward(self, x):
        # siz_batch, num_token, _ = x.shape
        shape = x.shape
        x = x.view(-1, self.dim_feature)
        x = self.norm(x)
        x = x.view(shape)
        return x


class MultiLayerPerceptron(AutoModule):
    def __init__(self, dim_seq, drop_out=0., bias=True,
                 act_func_hidden=nn.GELU, act_func_output=nn.Identity, **kwargs):
        """:param dim_seq: A Iterable that defining all layer dim from input to output (in, hid1, hid2..., out)."""
        super().__init__(locals())

        assert len(dim_seq) >= 2

        self.layers = nn.Sequential()
        for layer in range(1, len(dim_seq)):
            self.layers += nn.Sequential(*[
                nn.Linear(dim_seq[layer - 1], dim_seq[layer], bias=bias),
                act_func_hidden() if layer < len(dim_seq) - 1 else act_func_output(),
                nn.Dropout(drop_out)
            ])

        # init weights
        for modules in self.modules():
            if isinstance(modules, nn.Linear):
                nn.init.normal_(modules.weight, std=0.002)
                if modules.bias is not None:
                    nn.init.constant_(modules.bias, 0)

    def forward(self, x):
        x = self.layers(x)
        return x


class SwiGLUChannelMixing(AutoModule):
    def __init__(self, dim_feature):
        super().__init__(locals())

        self.linear_w = nn.Linear(dim_feature, dim_feature, bias=False)
        self.linear_v = nn.Linear(dim_feature, dim_feature, bias=False)
        self.linear_o = nn.Linear(dim_feature, dim_feature, bias=False)
        for linear in [self.linear_w, self.linear_v, self.linear_o]:
            nn.init.orthogonal_(linear.weight)

    def forward(self, x):
        """:param: x: A tensor (siz_batch, num_token, dim_feature) """

        w = self.linear_w(x)
        v = self.linear_v(x)

        o = self.linear_o(w * torch.sigmoid(w) * v)

        return o











class SinCosPosEmbedding(AutoModule):
    def __init__(self, dim_embed, omega=10000, do='x->x', **kwargs):
        super().__init__(locals())

        self.namespace, self.returning = self.phase_flow(do)
        assert len(self.namespace) == len(self.returning)

        self.position = self.declare_auto_state('position')

        self.register_buffer('embed', torch.arange(0, dim_embed // 2).to(torch.float64))
        self.embed /= dim_embed / 2.
        self.embed = 1. / self.omega ** self.embed

    def forward(self, *args):
        """:param args: A tuple of tensor (... , num_token, dim_feature)"""
        num_token = args[0].shape[-2]

        position = self.position.if_not(init=lambda: 0)

        self.position.update(position + num_token)

        grid = torch.arange(position, position + num_token).to(self.embed)
        out = torch.einsum('g,o->go', grid, self.embed)
        # out (num_token, dim_embed / 2)

        emb = torch.cat((torch.sin(out), torch.cos(out)), dim=1).to(self.dtype)

        output = [emb + x for x in args]

        return output[0] if len(output) == 1 else output


class RandomPosEmbedding(AutoModule):
    def __init__(self, dim_embed, max_position, do='x->x'):
        super().__init__(locals())

        self.namespace, self.returning = self.phase_flow(do)
        assert len(self.namespace) == len(self.returning)

        self.embedding = nn.Embedding(max_position, dim_embed)

        nn.init.normal_(self.embedding.weight, std=0.02)

        self.position = self.declare_auto_state('position')

    def forward(self, *args):
        """:param args: A tuple of tensor (... , num_token, dim_feature)"""
        num_token = args[0].shape[-2]

        position = self.position.if_not(init=lambda: 0)

        self.position.update(position + num_token)

        assert position + num_token <= self.max_position

        emb = torch.arange(num_token, dtype=torch.long, device=self.device) + position
        emb = self.embedding(emb)

        output = [emb + x for x in args]

        return output[0] if len(output) == 1 else output









class WeightedCumSumCuda(AutoKernel):
    pass


# from auto import *

from functools import partial
from functools import partial as p

# from ..arch import TimeChannelMixingBlock, TokenShift, GroupNorm


class WKV5Cuda(AutoKernel):
    def __init__(self, dim_feature, num_head,
                 type_data=torch.bfloat16, type_cuda=torch.float32, type_u_w=torch.float32):
        super().__init__(locals(), self.kernel_forward, self.kernel_backward)

        assert dim_feature % num_head == 0
        self.siz_head = dim_feature // num_head

        url = os.path.join(os.path.dirname(__file__), 'wkv5.cu')

        self.type_data_name = AutoCudaKernel.to_ctype(type_data)
        self.type_cuda_name = AutoCudaKernel.to_ctype(type_cuda)
        self.type_u_w_name = AutoCudaKernel.to_ctype(type_u_w)

        self.cuda = AutoCudaKernel.link(url=url,
                                        forward='cuda_forward',
                                        backward='cuda_backward',
                                        flags=[
                                            '-D' + f'AUTO_CUDA_KERNEL_PRE_DEFINE="'
                                                   f'typedef {self.type_data_name} ftype;'
                                                   f'typedef {self.type_cuda_name} ctype;'
                                                   f'typedef {self.type_u_w_name} utype;'
                                                   f'constexpr int siz_head = {self.siz_head};'
                                                   f'"'
                                        ])

    def kernel_forward(self, ctx, r, k, v, u, w, s):
        u, w = map(lambda x: x.type(self.type_u_w), (u, w))
        r, k, v, s = map(lambda x: x.type(self.type_data), (r, k, v, s))
        r, k, v, u, w, s = map(lambda x: x.contiguous(), (r, k, v, u, w, s))

        siz_batch, num_token, _ = r.shape

        ctx.save_for_backward(r, k, v, u, w, s)

        new = partial(torch.zeros, dtype=self.type_data, device=r.device)
        y = new((siz_batch, num_token, self.dim_feature))
        s = s.clone().detach()

        self.cuda.forward(siz_batch, num_token, self.dim_feature, self.num_head,
                          r, k, v, u, w,
                          y, s)

        return y, s

    def kernel_backward(self, ctx, gy, gs):
        gy, gs = map(lambda x: x.type(self.type_data), (gy, gs))
        gy, gs = map(lambda x: x.contiguous(), (gy, gs))

        siz_batch, num_token, _ = gy.shape

        r, k, v, u, w, s = ctx.saved_tensors

        new = partial(torch.zeros, dtype=self.type_data, device=gy.device)
        gr = new((siz_batch, num_token, self.dim_feature))
        gk = new((siz_batch, num_token, self.dim_feature))
        gv = new((siz_batch, num_token, self.dim_feature))
        gu = new((self.num_head, self.siz_head), dtype=self.type_u_w)
        gw = new((self.num_head, self.siz_head), dtype=self.type_u_w)
        gs = gs.clone()

        self.cuda.backward(siz_batch, num_token, self.dim_feature, self.num_head,
                           r, k, v, u, w, s,
                           gy, gs,
                           gr, gk, gv, gu, gw)

        return gr, gk, gv, gu, gw, gs

    def naive_style(self, r, k, v, u, w, s):
        siz_batch, num_token, _ = r.shape

        shape = siz_batch, num_token, self.num_head, self.siz_head
        r, k, v = map(lambda arg: arg.view(*shape).permute(0, 2, 1, 3), (r, k, v))
        u = u.view(1, self.num_head, self.siz_head, 1)
        w = w.view(1, self.num_head, self.siz_head, 1)

        mix = []
        for t in range(num_token):
            rr, kk, vv = map(lambda arg: arg[:, :, t:t + 1, :], (r, k, v))

            kv = kk.transpose(-1, -2) @ vv
            yy = rr @ (u * kv + s)
            s = w * s + kv

            mix.append(yy)

        mix = torch.cat(mix, dim=-2)
        mix = mix.permute(0, 2, 1, 3).reshape(siz_batch, num_token, self.dim_feature)

        return mix, s

    @classmethod
    def check_grad(cls):
        siz_batch, num_token, dim_feature, num_head = 2, 12, 128, 2
        siz_head = dim_feature // num_head

        r = torch.randn(siz_batch, num_token, dim_feature)
        k = torch.randn(siz_batch, num_token, dim_feature)
        v = torch.randn(siz_batch, num_token, dim_feature)
        u = torch.randn(num_head, siz_head)
        w = torch.rand(num_head, siz_head)
        s = torch.randn(siz_batch, num_head, siz_head, siz_head)

        kernel = cls(dim_feature, num_head, type_data=torch.bfloat16, type_cuda=torch.float32, type_u_w=torch.float32)
        print('::: check diff in bf16 ')
        AutoKernel.diff_check(kernel, kernel.naive_style, [r, k, v, u, w, s], 'rkvuws', 'ys', dtype=torch.bfloat16)
        print('::: check diff in bf16 -- fp64')
        AutoKernel.diff_check(kernel, kernel.naive_style, [r, k, v, u, w, s], 'rkvuws', 'ys', dtype=torch.float64)

        kernel = cls(dim_feature, num_head, type_data=torch.float64, type_cuda=torch.float64, type_u_w=torch.float64)
        print('::: check diff in fp64 ')
        AutoKernel.diff_check(kernel, kernel.naive_style, [r, k, v, u, w, s], 'rkvuws', 'ys', dtype=torch.float64)
        print('::: check grad in fp64 ')
        AutoKernel.grad_check(kernel, [r, k, v, u, w, s])


class RWKV5TimeMixing(AutoModule):
    def __init__(self, dim_feature, num_head, normalizer=None, w_init_std=0.2, u_init_std=0.2, **kwargs):
        super().__init__(locals())

        assert dim_feature % num_head == 0
        self.siz_head = dim_feature // num_head

        self.wkv_kernel = WKV5Cuda(dim_feature, num_head)

        self.linear_r = nn.Linear(self.dim_feature, self.dim_feature, bias=False)
        self.linear_k = nn.Linear(self.dim_feature, self.dim_feature, bias=False)
        self.linear_v = nn.Linear(self.dim_feature, self.dim_feature, bias=False)
        self.linear_g = nn.Linear(self.dim_feature, self.dim_feature, bias=False)
        self.linear_o = nn.Linear(self.dim_feature, self.dim_feature, bias=False)
        for linear in [self.linear_r, self.linear_k, self.linear_v, self.linear_g, self.linear_o]:
            nn.init.orthogonal_(linear.weight)

        self.decay_u = nn.Parameter(nn.init.normal_(torch.zeros(self.num_head, self.siz_head), std=u_init_std))
        self.decay_w = nn.Parameter(nn.init.normal_(torch.zeros(self.num_head, self.siz_head), std=w_init_std))

        self.normalizer = normalizer and self.instantiate(normalizer) or nn.LayerNorm(dim_feature)

        self.state_s = self.declare_auto_state('hidden_state')
        # hidden_state (siz_batch, num_head, siz_head, siz_head)

    def forward(self, r, k, v, g):
        """:param: r k v g: A tensor (siz_batch, num_token, dim_feature)"""

        siz_batch, num_token, _ = r.shape

        r, k, v, g = self.linear_r(r), self.linear_k(k), self.linear_v(v), self.linear_g(g)
        # r k v g (siz_batch, num_token, dim_feature)

        s = self.state_s.if_not(
            init=lambda: torch.zeros(siz_batch, self.num_head, self.siz_head, self.siz_head).to(r)
        ).detach()

        u, w = self.decay_u, torch.exp(-torch.exp(self.decay_w.float()))

        x, s = self.wkv_kernel(r, k, v, u, w, s)

        self.state_s.update(s)

        x = self.normalizer(x.view(siz_batch * num_token, self.dim_feature))
        x = x.view(siz_batch, num_token, self.dim_feature)
        # x (siz_batch, num_token, dim_feature)

        x = self.linear_o(x * nn.functional.silu(g))

        return x


class RWKV5ChannelMixing(AutoModule):
    def __init__(self, dim_feature, act_func_r=None, act_func_k=None):
        super().__init__(locals())

        self.linear_r = nn.Linear(dim_feature, dim_feature, bias=False)
        self.linear_k = nn.Linear(dim_feature, dim_feature, bias=False)
        self.linear_v = nn.Linear(dim_feature, dim_feature, bias=False)

        for linear in [self.linear_r, self.linear_k, self.linear_v]:
            nn.init.orthogonal_(linear.weight)

        self.act_func_r = act_func_r or torch.sigmoid
        self.act_func_k = act_func_k or torch.relu

    def forward(self, r, k):
        """:param: r k: A tensor (siz_batch, num_token, dim_feature) """

        r = torch.sigmoid(self.linear_r(r))
        k = torch.square(torch.relu(self.linear_k(k)))
        x = self.linear_v(k) * r

        return x


class RWKV1ChannelMixing(AutoModule):
    def __init__(self, dim_feature, dim_hidden=None, act_func_key=torch.nn.ReLU):
        super().__init__(locals())

        self.dim_hidden = dim_hidden or dim_feature

        self.linear_r = nn.Linear(dim_feature, dim_feature, bias=False)
        self.linear_k = nn.Linear(dim_feature, self.dim_hidden, bias=False)
        self.linear_v = nn.Linear(self.dim_hidden, dim_feature, bias=False)

        self.act_func = act_func_key()

        # init weights
        for linear in [self.linear_r, self.linear_k, self.linear_v]:
            nn.init.orthogonal_(linear.weight)

    def forward(self, x):
        k = self.act_func(self.linear_k(x)) ** 2
        x = torch.sigmoid(self.linear_r(x)) * self.linear_v(k)
        return x


# from auto import *

from functools import partial
from functools import partial as p

# from ..arch import TimeChannelMixingBlock, TokenShift, SwiGLUChannelMixing, RWKV5ChannelMixing, GroupNorm


class GLACuda(AutoKernel):
    def __init__(self, dim_feature, num_head, type_data=torch.bfloat16, type_cuda=torch.float32):
        super().__init__(locals(), self.kernel_forward, self.kernel_backward)

        assert dim_feature % num_head == 0
        self.siz_head = dim_feature // num_head

        url = os.path.join(os.path.dirname(__file__), 'gated_att.cu')

        self.type_data_name = AutoCudaKernel.to_ctype(type_data)
        self.type_cuda_name = AutoCudaKernel.to_ctype(type_cuda)

        self.cuda = AutoCudaKernel.link(url=url,
                                        forward='cuda_forward',
                                        backward='cuda_backward',
                                        flags=[
                                            '-D' + f'AUTO_CUDA_KERNEL_PRE_DEFINE="'
                                                   f'typedef {self.type_data_name} ftype;'
                                                   f'typedef {self.type_cuda_name} ctype;'
                                                   f'constexpr int siz_head = {self.siz_head};'
                                                   f'"'
                                        ])

    def kernel_forward(self, ctx, q, k, g, v, s):
        q, k, g, v, s = map(lambda x: x.type(self.type_data), (q, k, g, v, s))
        q, k, g, v, s = map(lambda x: x.contiguous(), (q, k, g, v, s))

        siz_batch, num_token, _ = q.shape

        ctx.save_for_backward(q, k, g, v, s)

        new = partial(torch.zeros, dtype=self.type_data, device=q.device)
        y = new((siz_batch, num_token, self.dim_feature))
        s = s.clone().detach()

        self.cuda.forward(siz_batch, num_token, self.dim_feature, self.num_head,
                          q, k, g, v, y, s)

        return y, s

    def kernel_backward(self, ctx, gy, gs):
        gy, gs = map(lambda x: x.type(self.type_data), (gy, gs))
        gy, gs = map(lambda x: x.contiguous(), (gy, gs))

        siz_batch, num_token, _ = gy.shape

        q, k, g, v, s = ctx.saved_tensors

        new = partial(torch.zeros, dtype=self.type_data, device=gy.device)

        gq = new((siz_batch, num_token, self.dim_feature))
        gk = new((siz_batch, num_token, self.dim_feature))
        gg = new((siz_batch, num_token, self.dim_feature))
        gv = new((siz_batch, num_token, self.dim_feature))
        gs = gs.clone()

        self.cuda.backward(siz_batch, num_token, self.dim_feature, self.num_head,
                           q, k, g, v, s,
                           gy, gs,
                           gq, gk, gg, gv)

        return gq, gk, gg, gv, gs

    def naive_style(self, q, k, g, v, s):
        siz_batch, num_token, _ = q.shape

        shape = siz_batch, num_token, self.num_head, self.siz_head
        q, k, g, v = map(lambda arg: arg.view(*shape).permute(0, 2, 1, 3), (q, k, g, v))

        mix = []
        for t in range(num_token):
            qq, kk, gg, vv = map(lambda arg: arg[:, :, t:t + 1, :], (q, k, g, v))

            s = gg.transpose(-1, -2) * s + kk.transpose(-1, -2) @ vv
            yy = qq @ s

            mix.append(yy)

        mix = torch.cat(mix, dim=-2)
        mix = mix.permute(0, 2, 1, 3).reshape(siz_batch, num_token, self.dim_feature)

        return mix, s

    @classmethod
    def check_grad(cls):
        siz_batch, num_token, dim_feature, num_head = 2, 12, 128, 2
        siz_head = dim_feature // num_head

        q = torch.randn(siz_batch, num_token, dim_feature)
        k = torch.randn(siz_batch, num_token, dim_feature)
        g = torch.rand(siz_batch, num_token, dim_feature)
        v = torch.randn(siz_batch, num_token, dim_feature)
        s = torch.randn(siz_batch, num_head, siz_head, siz_head)

        kernel = cls(dim_feature, num_head, type_data=torch.bfloat16, type_cuda=torch.float32)
        print('::: check diff in bf16 ')
        AutoKernel.diff_check(kernel, kernel.naive_style, [q, k, g, v, s], 'qkgvs', 'ys', dtype=torch.bfloat16)
        print('::: check diff in bf16 -- fp64 ')
        AutoKernel.diff_check(kernel, kernel.naive_style, [q, k, g, v, s], 'qkgvs', 'ys', dtype=torch.float64)

        kernel = cls(dim_feature, num_head, type_data=torch.float64, type_cuda=torch.float64)
        print('::: check diff in fp64 ')
        AutoKernel.diff_check(kernel, kernel.naive_style, [q, k, g, v, s], 'qkgvs', 'ys', dtype=torch.float64)
        print('::: check grad in fp64 ')
        AutoKernel.grad_check(kernel, [q, k, g, v, s])


class GatedLinearAttention(AutoModule):
    def __init__(self, dim_feature, num_head, normalizer=None, emb_position=None, **kwargs):
        super().__init__(locals())

        assert dim_feature % num_head == 0
        self.siz_head = dim_feature // num_head

        self.kernel = GLACuda(dim_feature, num_head)

        self.linear_q = nn.Linear(dim_feature, dim_feature, bias=False)
        self.linear_k = nn.Linear(dim_feature, dim_feature, bias=False)
        self.linear_v = nn.Linear(dim_feature, dim_feature, bias=False)

        self.linear_g = nn.Sequential(
            nn.Linear(dim_feature, 16, bias=False),
            nn.Linear(16, dim_feature, bias=True),
        )

        self.linear_r = nn.Linear(dim_feature, dim_feature, bias=True)
        self.linear_o = nn.Linear(dim_feature, dim_feature, bias=False)

        self.normalizer = normalizer and self.instantiate(normalizer) or GroupNorm(dim_feature, num_head)

        self.emb_position = emb_position and self.instantiate(emb_position, dim_embed=self.siz_head) or AutoIdentity()

        self.state = self.declare_auto_state('hidden_state')
        # hidden_state (siz_batch, num_head, siz_head, siz_head)

        # init weights
        for linear in [self.linear_q, self.linear_k, self.linear_v, self.linear_o]:
            nn.init.orthogonal_(linear.weight)

        for linear in [self.linear_r, self.linear_g[0], self.linear_g[1]]:
            nn.init.normal_(linear.weight, std=0.002)
            if linear.bias is not None:
                nn.init.constant_(linear.bias, 0)

    def forward(self, x, q, k, v, g):

        siz_batch, num_token, _ = x.shape

        q, k, v, g = self.linear_q(q), self.linear_k(k), self.linear_v(v), self.linear_g(g)
        # q k v g (siz_batch, num_token, dim_feature)

        g = torch.sigmoid(g)

        k = self.emb_position(k)

        s = self.state.if_not(
            init=lambda: torch.zeros(siz_batch, self.num_head, self.siz_head, self.siz_head).to(q)
        ).detach()

        mix, s = self.kernel(q, k, g, v, s)
        # mix (siz_batch, num_token, dim_feature)
        # s (siz_batch, num_head, siz_head, siz_head)

        self.state.update(s)

        x = self.linear_r(x)
        mix = self.normalizer(mix)
        x = self.linear_o(mix * x * torch.sigmoid(x))
        # x (siz_batch, num_token, dim_feature)

        return x


def block_default_gla__(dim_feature, num_head):
    """Default GLA block config"""
    return p(TimeChannelMixingBlock,
             dim_feature=dim_feature,
             num_head=num_head,
             normalizer=p(GroupNorm, num_head=1),
             time_mixing=p(AutoSequential, do='x->x',
                           module_list=[
                               p(AutoSum, do='x -> q, k, v, g'),
                               p(GatedLinearAttention)
                           ]),
             channel_mixing=p(AutoSequential, do='x->x',
                              module_list=[
                                  p(SwiGLUChannelMixing)
                              ]),
             )()


def block_gla_with_token_shift__(dim_feature, num_head):
    """A better GLA block config, but slower"""
    return p(TimeChannelMixingBlock,
             dim_feature=dim_feature,
             num_head=num_head,
             normalizer=p(GroupNorm, num_head=1),
             time_mixing=p(AutoSequential, do='x->x',
                           module_list=[
                               p(TokenShift, do='x -> shift'),
                               p(AutoSum, do='shift -> q, k, v'),
                               p(AutoSum, do='x -> g'),
                               p(GatedLinearAttention,
                                 normalizer=p(nn.LayerNorm, normalized_shape=dim_feature),
                                 emb_position=None,
                                 )
                           ]),
             channel_mixing=p(AutoSequential, do='x->x',
                              module_list=[
                                  p(TokenShift, do='x -> r, k'),
                                  p(RWKV5ChannelMixing)
                              ]),
             )()




