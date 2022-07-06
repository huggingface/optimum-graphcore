from transformers.models.convnext.modeling_convnext import ConvNextLayer


class OptimizedConvNextLayer(ConvNextLayer):
    def forward(self, hidden_states):
        """
        Merge the 2nd and 3rd dimensions of the tensor before pwconv, and restore the shape afterwards.
        This is because currently, nn.Linear() does not work efficiently on 4-dimensional inputs.
        """
        input = hidden_states
        x = self.dwconv(hidden_states)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.layernorm(x)
        N, H, W, C = x.shape
        # Reshape for running efficiently on IPUs
        x = x.view(N, -1, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # Restore the shape
        x = x.view(N, H, W, C)
        if self.layer_scale_parameter is not None:
            x = self.layer_scale_parameter * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
