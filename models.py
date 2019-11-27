import torch
import torch.nn


class ConvolutionalAutoencoder(torch.nn.Module):

    def __init__(self, n_blocks, downsampling_method, upsampling_method,
                 layers_per_block=2):
        super().__init__()
        self.n_blocks = n_blocks
        self.downsampling_method = downsampling_method
        self.upsampling_method = upsampling_method

        self.encoder = ConvolutionalEncoder(n_blocks, downsampling_method,
                                            layers_per_block)
        self.decoder = ConvolutionalDecoder(n_blocks, upsampling_method,
                                            self.encoder.output_channels,
                                            layers_per_block)

    def forward(self, x):
        code = self.encoder(x)
        reconstruction = self.decoder(code)
        return reconstruction


class ConvolutionalEncoder(torch.nn.Module):

    def __init__(self, n_blocks, downsampling_method, layers_per_block=2,
                 init_filters=16, kernel_size=5, input_channels=1):
        super().__init__()
        self.n_blocks = n_blocks
        self.downsampling_method = downsampling_method
        self.layers_per_block = layers_per_block
        self.init_filters = init_filters
        self.kernel_size = kernel_size
        self.input_channels = input_channels

        layers = []

        # First layer so we have <init_filters> channels.
        n_filters = init_filters
        layers.append(torch.nn.Conv2d(input_channels, n_filters, kernel_size,
                                      padding=kernel_size // 2))
        layers.append(torch.nn.ReLU())

        # Encoding blocks.
        for _ in range(n_blocks):
            layers.append(ConvolutionalBlock(n_filters, kernel_size, layers_per_block))
            # Double the number of filters.
            n_filters = 2 * n_filters
            # Down-sampling
            # TODO

        self.encoder = torch.nn.Sequential(*layers)
        self.output_channels = n_filters

    def forward(self, x):
        return self.encoder(x)


class ConvolutionalDecoder(torch.nn.Module):

    def __init__(self, n_blocks, upsampling_method, init_filters,
                 layers_per_block=2, kernel_size=5, output_channels=1):
        super().__init__()
        self.n_blocks = n_blocks
        self.upsampling_method = upsampling_method
        self.layers_per_block = layers_per_block
        self.init_filters = init_filters
        self.kernel_size = kernel_size
        self.output_channels = output_channels

        layers = []
        # Decoding blocks.
        n_filters = init_filters
        for _ in range(n_blocks):
            layers.append(ConvolutionalBlock(n_filters, kernel_size, layers_per_block))
            # Double the number of filters.
            n_filters = n_filters // 2
            # Up-sampling
            # TODO

        # Last layer so we have <output_channels> channel.
        layers.append(torch.nn.Conv2d(n_filters, output_channels, kernel_size,
                                      padding=kernel_size // 2))
        layers.append(torch.nn.Tanh())

        self.decoder = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class ConvolutionalBlock(torch.nn.Module):
    """

    Applies n_layers convolutional layers with the same number of
    filters and filter sizes with ReLU activations
    keeping the same spacial size.
    """

    def __init__(self, n_filters, kernel_size, n_layers, last_stride=1):
        super().__init__()
        layers = []
        padding = kernel_size // 2  # To keep the same size.

        # First convolutional layers (Conv + ReLU).
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Conv2d(n_filters, n_filters,
                                          kernel_size=kernel_size,
                                          padding=padding))
            layers.append(torch.nn.ReLU())

        # Last convolutional layer (Strided-Conv + ReLU).
        layers.append(torch.nn.Conv2d(n_filters, n_filters,
                                      kernel_size, last_stride, padding))
        layers.append(torch.nn.ReLU())

        # To sequentially apply the layers.
        self.block = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
