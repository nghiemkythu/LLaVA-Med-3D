import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}

class MLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        return self.proj(x)

class DownSampleBlock(nn.Module):

        def forward(self, x):
            vit_embeds = x
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.flat_square(vit_embeds)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            return vit_embeds

        def flat_square(self, x):
            n, w, h, c = x.size()
            if w % 2 == 1:
                x = torch.concat([x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
                n, w, h, c = x.size()
            if h % 2 == 1:
                x = torch.concat([x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
                n, w, h, c = x.size()
            x = x.view(n, w, int(h / 2), int(c * 2))
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
            return x

class ResidualBlock(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

class RedesignedMLPBlock(nn.Module):
    def __init__(self, mm_channels, channels, mlp_depth):
        super().__init__()
        #self.module = module
        self.pre_norm = nn.LayerNorm(mm_channels)
        self.linear = nn.Linear(mm_channels, channels)
        self.mlp = MLP(channels) 
        self.relu = nn.ReLU()
        self.mlp_depth = mlp_depth 

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.linear(x)
        x0 = x 
        for _ in range(1, self.mlp_depth):
            x = self.mlp(x)
        return x0 + self.relu(x)

class SimpleResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.pre_norm = nn.LayerNorm(channels)

            self.proj = nn.Sequential(
                    nn.Linear(channels, channels), 
                    nn.GELU(), 
                    nn.Linear(channels, channels))

        def forward(self, x):
            x = self.pre_norm(x)
            return x + self.proj(x)

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    
    if 'res' in projector_type:
        mlp_gelu_resnet_match = re.match(r"^mlp(\d+)x_res(\d+)x_gelu$", projector_type)
        if mlp_gelu_resnet_match:
            mlp_depth = int(mlp_gelu_resnet_match.group(1))
            res_depth = int(mlp_gelu_resnet_match.group(2))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)] 
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            for _ in range(res_depth):
                modules.append(SimpleResBlock(config.hidden_size))
            return nn.Sequential(*modules)
    elif 'redesigned' in projector_type:
        mlp_gelu_red_match = re.match(r'^mlp(\d+)x_gelu_redesigned', projector_type)
        if mlp_gelu_red_match:
            mlp_depth = int(mlp_gelu_red_match.group(1))
            print('MLP Redesigned depth: ',mlp_depth)
            modules = RedesignedMLPBlock(config.mm_hidden_size, config.hidden_size, mlp_depth)
            return nn.Sequential(modules)
    else:
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            print('MLP depth: ',mlp_depth)
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.Sigmoid())
                print(config.hidden_size, config.hidden_size)
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return_model = nn.Sequential(*modules)

            for n, param in return_model.named_parameters():
                print(n, param.shape)
            print("========================")
            for mo in modules:
                for n, param in mo.named_parameters():
                    print(n, param.shape)

            return return_model
     
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
