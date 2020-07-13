import torch
import torch.nn as nn

def conv_batchNorm(in_channels, out_channels):
	return nn.Sequential(
		nn.Conv3d(in_channels, out_channels, 3, padding=1),
		nn.BatchNorm3d(out_channels),
		nn.ReLU(inplace=True),
		nn.Conv3d(out_channels, out_channels, 3, padding=1),
		nn.BatchNorm3d(out_channels),
		nn.ReLU(inplace=True),
	)   

def transition_layer(in_channels, out_channels):
	return nn.Sequential(
		nn.Conv3d(in_channels, out_channels, 3, padding=1),
		nn.BatchNorm3d(out_channels),
		nn.ReLU(inplace=True)
	)

class encoder(nn.Module):
	def __init__(self, n_class=1, ngf=32):
		super().__init__()

		self.dconv_down1 = conv_batchNorm(1, ngf)
		self.dconv_down2 = conv_batchNorm(ngf, ngf*2)
		self.dconv_down3 = conv_batchNorm(ngf*2, ngf*4)
		self.dconv_down4 = conv_batchNorm(ngf*4, ngf*8)

		self.avgpool = nn.AvgPool3d(2)

		self.transition1 = transition_layer(ngf, ngf//4)
		self.transition2 = transition_layer(ngf*2, ngf//2)
		self.transition3 = transition_layer(ngf*4, ngf)
		
		
	def forward(self, x):
		transition_list = []

		conv1 = self.dconv_down1(x)
		x = self.avgpool(conv1)
		tran1 = self.transition1(conv1)
		transition_list.append(tran1)

		conv2 = self.dconv_down2(x)
		x = self.avgpool(conv2)
		tran2 = self.transition2(conv2)
		transition_list.append(tran2)
		
		conv3 = self.dconv_down3(x)
		x = self.avgpool(conv3)
		tran3 = self.transition3(conv3)
		transition_list.append(tran3)
		
		x = self.dconv_down4(x)
		
		return x, transition_list

class decoder(nn.Module):
	def __init__(self, n_class, ngf=32):
		super().__init__()

		self.upsample3 = nn.ConvTranspose3d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up3 = conv_batchNorm(ngf + ngf*8, ngf*4)

		self.upsample2 = nn.ConvTranspose3d(ngf*4, ngf*4, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up2 = conv_batchNorm(ngf//2 + ngf*4, ngf*2)

		self.upsample1 = nn.ConvTranspose3d(ngf*2, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up1 = conv_batchNorm(ngf//4 + ngf*2, ngf)
		
		self.conv_last = nn.Conv3d(ngf, n_class, 1)
		
		
	def forward(self, x, transition_list):
		x = self.upsample3(x)
		x = torch.cat([x, transition_list[-1]], dim=1)
		
		x = self.dconv_up3(x)
		x = self.upsample2(x)        
		x = torch.cat([x, transition_list[-2]], dim=1)

		x = self.dconv_up2(x)
		x = self.upsample1(x)        
		x = torch.cat([x, transition_list[-3]], dim=1)
		
		x = self.dconv_up1(x)

		out = self.conv_last(x)
		
		return out

class encoder_decoer(nn.Module):
	def __init__(self, n_class, ngf=32):
		super().__init__()
		self.encoder = encoder(n_class, ngf)
		self.decoder = decoder(n_class, ngf)

	def forward(self, x):
		x, transition_list = self.encoder(x)
		x = self.decoder(x, transition_list)

		return x

class transition_UNet_small(nn.Module):

	def __init__(self, n_class, ngf=32):
		super().__init__()
				
		self.dconv_down1 = conv_batchNorm(1, ngf)
		self.dconv_down2 = conv_batchNorm(ngf, ngf*2)
		self.dconv_down3 = conv_batchNorm(ngf*2, ngf*4)    

		self.avgpool = nn.AvgPool3d(2)

		self.transition1 = transition_layer(ngf, ngf//4)
		self.transition2 = transition_layer(ngf*2, ngf//2)

		self.upsample2 = nn.ConvTranspose3d(ngf*4, ngf*4, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up2 = conv_batchNorm(ngf//2 + ngf*4, ngf*2)

		self.upsample1 = nn.ConvTranspose3d(ngf*2, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up1 = conv_batchNorm(ngf//4 + ngf*2, ngf)
		
		self.conv_last = nn.Conv3d(ngf, n_class, 1)
		
		
	def forward(self, x, return_features=False):
		conv1 = self.dconv_down1(x)
		x = self.avgpool(conv1)
		tran1 = self.transition1(conv1)

		conv2 = self.dconv_down2(x)
		x = self.avgpool(conv2)
		tran2 = self.transition2(conv2)
		
		x = self.dconv_down3(x)

		x = self.upsample2(x)        
		x = torch.cat([x, tran2], dim=1)       

		x = self.dconv_up2(x)
		x = self.upsample1(x)        
		x = torch.cat([x, tran1], dim=1)   
		
		last_features = self.dconv_up1(x)

		if return_features:
			out = last_features
		else:
			out = self.conv_last(last_features)
		
		return out

class transition_UNet(nn.Module):

	def __init__(self, n_class, ngf=32):
		super().__init__()
				
		self.dconv_down1 = conv_batchNorm(1, ngf)
		self.dconv_down2 = conv_batchNorm(ngf, ngf*2)
		self.dconv_down3 = conv_batchNorm(ngf*2, ngf*4)
		self.dconv_down4 = conv_batchNorm(ngf*4, ngf*8)        

		self.avgpool = nn.AvgPool3d(2)

		self.transition1 = transition_layer(ngf, ngf//4)
		self.transition2 = transition_layer(ngf*2, ngf//2)
		self.transition3 = transition_layer(ngf*4, ngf)

		self.upsample3 = nn.ConvTranspose3d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up3 = conv_batchNorm(ngf + ngf*8, ngf*4)

		self.upsample2 = nn.ConvTranspose3d(ngf*4, ngf*4, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up2 = conv_batchNorm(ngf//2 + ngf*4, ngf*2)

		self.upsample1 = nn.ConvTranspose3d(ngf*2, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up1 = conv_batchNorm(ngf//4 + ngf*2, ngf)
		
		self.conv_last = nn.Conv3d(ngf, n_class, 1)
		
		
	def forward(self, x, return_features=False):
		conv1 = self.dconv_down1(x)
		x = self.avgpool(conv1)
		tran1 = self.transition1(conv1)

		conv2 = self.dconv_down2(x)
		x = self.avgpool(conv2)
		tran2 = self.transition2(conv2)
		
		conv3 = self.dconv_down3(x)
		x = self.avgpool(conv3)
		tran3 = self.transition3(conv3)
		
		x = self.dconv_down4(x)
		x = self.upsample3(x)
		x = torch.cat([x, tran3], dim=1)
		
		x = self.dconv_up3(x)
		x = self.upsample2(x)        
		x = torch.cat([x, tran2], dim=1)       

		x = self.dconv_up2(x)
		x = self.upsample1(x)        
		x = torch.cat([x, tran1], dim=1)   
		
		last_features = self.dconv_up1(x)
		
		# out = self.conv_last(last_features)
		# return out, last_features

		if return_features:
			out = last_features
		else:
			out = self.conv_last(last_features)
		
		return out

class transition_UNet_large(nn.Module):

	def __init__(self, n_class, ngf=32):
		super().__init__()
				
		self.dconv_down1 = conv_batchNorm(1, ngf)
		self.dconv_down2 = conv_batchNorm(ngf, ngf*2)
		self.dconv_down3 = conv_batchNorm(ngf*2, ngf*4)
		self.dconv_down4 = conv_batchNorm(ngf*4, ngf*8)
		self.dconv_down5 = conv_batchNorm(ngf*8, ngf*16)

		self.avgpool = nn.AvgPool3d(2)

		self.transition1 = transition_layer(ngf, ngf//8)
		self.transition2 = transition_layer(ngf*2, ngf//4)
		self.transition3 = transition_layer(ngf*4, ngf//2)
		self.transition4 = transition_layer(ngf*8, ngf)

		self.upsample4 = nn.ConvTranspose3d(ngf*16, ngf*16, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up4 = conv_batchNorm(ngf + ngf*16, ngf*8)

		self.upsample3 = nn.ConvTranspose3d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up3 = conv_batchNorm(ngf//2 + ngf*8, ngf*4)

		self.upsample2 = nn.ConvTranspose3d(ngf*4, ngf*4, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up2 = conv_batchNorm(ngf//4 + ngf*4, ngf*2)

		self.upsample1 = nn.ConvTranspose3d(ngf*2, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up1 = conv_batchNorm(ngf//8 + ngf*2, ngf)
		
		self.conv_last = nn.Conv3d(ngf, n_class, 1)
		
		
	def forward(self, x, return_features=False):
		conv1 = self.dconv_down1(x)
		x = self.avgpool(conv1)
		tran1 = self.transition1(conv1)

		conv2 = self.dconv_down2(x)
		x = self.avgpool(conv2)
		tran2 = self.transition2(conv2)
		
		conv3 = self.dconv_down3(x)
		x = self.avgpool(conv3)
		tran3 = self.transition3(conv3)
		
		conv4 = self.dconv_down4(x)
		x = self.avgpool(conv4)
		tran4 = self.transition4(conv4)

		x = self.dconv_down5(x)
		x = self.upsample4(x)
		x = torch.cat([x, tran4], dim=1)
		
		x = self.dconv_up4(x)
		x = self.upsample3(x)        
		x = torch.cat([x, tran3], dim=1)       

		x = self.dconv_up3(x)
		x = self.upsample2(x)        
		x = torch.cat([x, tran2], dim=1)   

		x = self.dconv_up2(x)
		x = self.upsample1(x)        
		x = torch.cat([x, tran1], dim=1)   
		
		last_features = self.dconv_up1(x)
		
		# out = self.conv_last(last_features)
		# return out, last_features

		if return_features:
			out = last_features
		else:
			out = self.conv_last(last_features)
		
		return out

def double_conv(in_channels, out_channels):
	return nn.Sequential(
		nn.Conv3d(in_channels, out_channels, 3, padding=1),
		nn.ReLU(inplace=True),
		nn.Conv3d(out_channels, out_channels, 3, padding=1),
		nn.ReLU(inplace=True)
	)   

class UNet_small(nn.Module):
	def __init__(self, n_class, ngf=32):
		super().__init__()
				
		self.dconv_down1 = double_conv(1, ngf)
		self.dconv_down2 = double_conv(ngf, ngf*2)
		self.dconv_down3 = double_conv(ngf*2, ngf*4) 

		self.maxpool = nn.AvgPool3d(2)

		self.upsample2 = nn.ConvTranspose3d(ngf*4, ngf*4, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up2 = double_conv(ngf*2 + ngf*4, ngf*2)

		self.upsample1 = nn.ConvTranspose3d(ngf*2, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up1 = double_conv(ngf + ngf*2, ngf)
		
		self.conv_last = nn.Conv3d(ngf, n_class, 1)
		
		
	def forward(self, x, return_features=False):
		conv1 = self.dconv_down1(x)
		x = self.maxpool(conv1)

		conv2 = self.dconv_down2(x)
		x = self.maxpool(conv2)
		
		x = self.dconv_down3(x)
		
		x = self.upsample2(x) # output channel 256
		x = torch.cat([x, conv2], dim=1)       

		x = self.dconv_up2(x)
		x = self.upsample1(x)        
		x = torch.cat([x, conv1], dim=1)   
		
		last_features = self.dconv_up1(x)
		

		if return_features:
			out = last_features
		else:
			out = self.conv_last(last_features)
		
		return out

class UNet(nn.Module):
	def __init__(self, n_class, ngf=32):
		super().__init__()
				
		self.dconv_down1 = double_conv(1, ngf)
		self.dconv_down2 = double_conv(ngf, ngf*2)
		self.dconv_down3 = double_conv(ngf*2, ngf*4)
		self.dconv_down4 = double_conv(ngf*4, ngf*8)        

		self.maxpool = nn.AvgPool3d(2)

		self.upsample3 = nn.ConvTranspose3d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, output_padding=1)   
		self.dconv_up3 = double_conv(ngf*4 + ngf*8, ngf*4)

		self.upsample2 = nn.ConvTranspose3d(ngf*4, ngf*4, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up2 = double_conv(ngf*2 + ngf*4, ngf*2)

		self.upsample1 = nn.ConvTranspose3d(ngf*2, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up1 = double_conv(ngf + ngf*2, ngf)
		
		self.conv_last = nn.Conv3d(ngf, n_class, 1)
		
		
	def forward(self, x, return_features=False):
		# print(x.size())
		conv1 = self.dconv_down1(x)
		# print(conv1.size())
		x = self.maxpool(conv1)

		conv2 = self.dconv_down2(x)
		x = self.maxpool(conv2)
		
		conv3 = self.dconv_down3(x) # output channel 256
		x = self.maxpool(conv3)   
		
		x = self.dconv_down4(x)
		
		x = self.upsample3(x) # output channel 512?
		x = torch.cat([x, conv3], dim=1)
		
		x = self.dconv_up3(x) # output channel 256
		x = self.upsample2(x) # output channel 256
		x = torch.cat([x, conv2], dim=1)       

		x = self.dconv_up2(x)
		x = self.upsample1(x)        
		x = torch.cat([x, conv1], dim=1)   
		
		last_features = self.dconv_up1(x)
		

		if return_features:
			out = last_features
		else:
			out = self.conv_last(last_features)
		
		return out

class UNet_large(nn.Module):
	def __init__(self, n_class, ngf=32):
		super().__init__()
				
		self.dconv_down1 = double_conv(1, ngf)
		self.dconv_down2 = double_conv(ngf, ngf*2)
		self.dconv_down3 = double_conv(ngf*2, ngf*4)
		self.dconv_down4 = double_conv(ngf*4, ngf*8)        

		self.maxpool = nn.AvgPool3d(2)

		self.upsample3 = nn.ConvTranspose3d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, output_padding=1)   
		self.dconv_up3 = double_conv(ngf*4 + ngf*8, ngf*4)

		self.upsample2 = nn.ConvTranspose3d(ngf*4, ngf*4, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up2 = double_conv(ngf*2 + ngf*4, ngf*2)

		self.upsample1 = nn.ConvTranspose3d(ngf*2, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.dconv_up1 = double_conv(ngf + ngf*2, ngf)
		
		self.conv_last = nn.Conv3d(ngf, n_class, 1)
		
		
	def forward(self, x, return_features=False):
		# print(x.size())
		conv1 = self.dconv_down1(x)
		# print(conv1.size())
		x = self.maxpool(conv1)

		conv2 = self.dconv_down2(x)
		x = self.maxpool(conv2)
		
		conv3 = self.dconv_down3(x) # output channel 256
		x = self.maxpool(conv3)   
		
		x = self.dconv_down4(x)
		
		x = self.upsample3(x) # output channel 512?
		x = torch.cat([x, conv3], dim=1)
		
		x = self.dconv_up3(x) # output channel 256
		x = self.upsample2(x) # output channel 256
		x = torch.cat([x, conv2], dim=1)       

		x = self.dconv_up2(x)
		x = self.upsample1(x)        
		x = torch.cat([x, conv1], dim=1)   
		
		last_features = self.dconv_up1(x)
		

		if return_features:
			out = last_features
		else:
			out = self.conv_last(last_features)
		
		return out