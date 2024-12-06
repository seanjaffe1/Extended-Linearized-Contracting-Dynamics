import torch
import logging

from manifold_flow import transforms
from manifold_flow.utils.various import product

logger = logging.getLogger(__name__)


class ProjectionSplit(transforms.Transform):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_dim_total = product(input_dim)
        self.output_dim_total = product(output_dim)
        self.mode_in = "vector" if isinstance(input_dim, int) else "image"
        self.mode_out = "vector" if isinstance(input_dim, int) else "image"

        logger.debug("Set up projection from %s with dimension %s to %s with dimension %s", self.mode_in, self.input_dim, self.mode_out, self.output_dim)

        assert self.input_dim_total >= self.output_dim_total, "Input dimension has to be larger than output dimension"

    def forward(self, inputs, **kwargs):
        if self.mode_in == "vector" and self.mode_out == "vector":
            u = inputs[:, : self.output_dim]
            rest = inputs[:, self.output_dim :]
        elif self.mode_in == "image" and self.mode_out == "vector":
            h = inputs.view(inputs.size(0), -1)
            u = h[:, : self.output_dim]
            rest = h[:, self.output_dim :]
        else:
            raise NotImplementedError("Unsuppoorted projection modes {}, {}".format(self.mode_in, self.mode_out))
        return u, rest

    def inverse(self, inputs, **kwargs):
        orthogonal_inputs = kwargs.get("orthogonal_inputs", torch.zeros(inputs.size(0), self.input_dim_total - self.output_dim))
        if self.mode_in == "vector" and self.mode_out == "vector":
            x = torch.cat((inputs, orthogonal_inputs), dim=1)
        elif self.mode_in == "image" and self.mode_out == "vector":
            c, h, w = self.input_dim
            x = torch.cat((inputs, orthogonal_inputs), dim=1)
            x = x.view(inputs.size(0), c, h, w)
        else:
            raise NotImplementedError("Unsuppoorted projection modes {}, {}".format(self.mode_in, self.mode_out))
        return x


class Projection(transforms.Transform):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_dim_total = product(input_dim)
        self.output_dim_total = product(output_dim)
        self.mode_in = "vector" if isinstance(input_dim, int) else "image"
        self.mode_out = "vector" if isinstance(input_dim, int) else "image"

        logger.debug("Set up projection from %s with dimension %s to %s with dimension %s", self.mode_in, self.input_dim, self.mode_out, self.output_dim)

        assert self.input_dim_total >= self.output_dim_total, "Input dimension has to be larger than output dimension"

    def forward(self, inputs, **kwargs):
        if self.mode_in == "vector" and self.mode_out == "vector":
            u = inputs[:, : self.output_dim]
        elif self.mode_in == "image" and self.mode_out == "vector":
            u = inputs.view(inputs.size(0), -1)
            u = u[:, : self.output_dim]
        else:
            raise NotImplementedError("Unsuppoorted projection modes {}, {}".format(self.mode_in, self.mode_out))
        return u

    def inverse(self, inputs, **kwargs):
        if self.mode_in == "vector" and self.mode_out == "vector":
            x = torch.cat((inputs, torch.zeros(inputs.size(0), self.input_dim - self.output_dim).to(inputs.device)), dim=1)
        elif self.mode_in == "image" and self.mode_out == "vector":
            c, h, w = self.input_dim
            x = torch.cat((inputs, torch.zeros(inputs.size(0), self.input_dim_total - self.output_dim)), dim=1)
            x = x.view(inputs.size(0), c, h, w)
        else:
            raise NotImplementedError("Unsuppoorted projection modes {}, {}".format(self.mode_in, self.mode_out))
        return x

class CompositeProjection(transforms.Transform):
    def __init__(self, composite_transform, input_dim, output_dim):
        super().__init__()
        self.transform = composite_transform
        self.projection = Projection(input_dim, output_dim)
        self.output_dim = output_dim


    def forward(self, inputs, context=None, full_jacobian=False):

        if full_jacobian:
            
            out, jac = self.transform(inputs, context=context, full_jacobian=full_jacobian)
            
            out = self.projection(out)
            return out, jac[:, :self.output_dim, :,]
        else:
            out, total_logabsdet = self.transform(inputs, context=context, full_jacobian=full_jacobian)
            out = self.projection(out)
            return out, total_logabsdet
    
    def inverse(self, inputs, context=None, full_jacobian=False):
        if full_jacobian:
            inputs = self.projection.inverse(inputs)
            out, jac = self.transform.inverse(inputs, context=context, full_jacobian=full_jacobian)
            

            return out, jac[:, :, :self.output_dim]
        else:
            inputs = self.projection.inverse(inputs)
            out, total_logabsdet = self.transform.inverse(inputs, context=context, full_jacobian=full_jacobian)

            return out, total_logabsdet