from torch import nn
import torch
import torch.nn.functional as F


class MyLinear(nn.Linear):
    # pass
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

        # print("init weight:")
        # print(self.weight)
        # asdfasdfadf

        # nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)




class MyNormalLinear(nn.Linear):
    # pass
    def reset_parameters(self) -> None:
        # nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MyPReLU(nn.Module):

    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1, init: float = 0.25,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()

        # use alpha instead of weight
        self.alpha = nn.parameter.Parameter(torch.empty(num_parameters, **factory_kwargs).fill_(init))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.prelu(input, self.alpha)

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)



if __name__ == "__main__":
    print("=" * 50)
    print("my_mlp 测试用例")
    print("=" * 50)

    torch.manual_seed(42)

    # 基本参数
    batch_size = 4
    in_features = 16
    hidden_features = 32
    out_features = 8

    # 构建层
    lin1 = MyLinear(in_features, hidden_features)
    act = MyPReLU(num_parameters=hidden_features)
    lin2 = MyNormalLinear(hidden_features, out_features)

    # 随机输入
    x = torch.randn(batch_size, in_features)
    print(f"输入形状: {x.shape}")

    # 前向传播
    h = lin1(x)
    h = act(h)
    y = lin2(h)
    print(f"输出形状: {y.shape}")

    # 简单损失与反向传播
    target = torch.randn_like(y)
    loss = F.mse_loss(y, target)
    print(f"Loss: {loss.item():.6f}")
    loss.backward()

    # 梯度检查
    grads = {
        'lin1.weight': lin1.weight.grad is not None,
        'lin1.bias': lin1.bias is not None and lin1.bias.grad is not None,
        'act.alpha': act.alpha.grad is not None,
        'lin2.weight': lin2.weight.grad is not None,
        'lin2.bias': lin2.bias is not None and lin2.bias.grad is not None,
    }
    print("梯度可用性：")
    for k, v in grads.items():
        print(f"  {k}: {v}")

    # 统计参数数量
    def count_params(m: nn.Module):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    total_params = count_params(lin1) + count_params(act) + count_params(lin2)
    print(f"可训练参数总数: {total_params}")

    print("=" * 50)
    print("测试完成")
    print("=" * 50)
