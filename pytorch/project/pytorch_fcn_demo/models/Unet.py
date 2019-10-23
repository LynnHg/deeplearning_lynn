import torch
import torch.nn as nn

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self. conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, n_class=10):
        super().__init__()

        # encode
        self.conv_encode1 = DoubleConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv_encode2 = DoubleConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv_encode3 = DoubleConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv_encode4 = DoubleConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv_encode5 = DoubleConvBlock(512, 1024)

        # decode
        self.conv_decode1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConvBlock(1024, 512)
        self.conv_decode2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConvBlock(512, 256)
        self.conv_decode3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConvBlock(256, 128)
        self.conv_decode4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConvBlock(128, 64)

        # classifier
        self.classifier = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # encode
        c1 = self.conv_encode1(x)
        p1 = self.pool1(c1)
        c2 = self.conv_encode2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv_encode3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv_encode4(p3)
        p4 = self.pool3(c4)
        c5 = self.conv_encode5(p4)

        # decode
        up1 = self.conv_decode1(c5)
        cat1 = torch.cat([up1, c4], dim=1)
        c6 = self.conv1(cat1)
        up2 = self.conv_decode2(c6)
        cat2 = torch.cat([up2, c3], dim=1)
        c7 = self.conv2(cat2)
        up3 = self.conv_decode3(c7)
        cat3 = torch.cat([up3, c2], dim=1)
        c8 = self.conv3(cat3)
        up4 = self.conv_decode4(c8)
        cat4 = torch.cat([up4, c1], dim=1)
        c9 = self.conv4(cat4)

        # up1 = self.conv_decode1(c5)
        # up1 += c4
        # up2 = self.conv_decode2(up1)
        # up2 += c3
        # up3 = self.conv_decode3(up2)
        # up3 += c2
        # up4 = self.conv_decode4(up3)
        # up4 += c1

        # classifier
        out = self.classifier(c9)

        return out

if __name__ == '__main__':
    print(Unet())