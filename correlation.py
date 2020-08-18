import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Correlation(nn.Module):
	def __init__(self,maxdisp):
		super(Correlation, self).__init__()

	self.maxdisp = maxdisp

	def forward(self, refimg_fea, targetimg_fea):
		"""
		another correlation function giving the same result as corr()
		supports backwards
		"""
		b,c,height,width = refimg_fea.shape
		if refimg_fea.is_cuda:
			cost = Variable(torch.cuda.FloatTensor(b,c,2*maxdisp+1,2*int(maxdisp//1)+1,height,width)).fill_(0.) # b,c,u,v,h,w
		else:
			cost = Variable(torch.FloatTensor(b,c,2*maxdisp+1,2*int(maxdisp//1)+1,height,width)).fill_(0.) # b,c,u,v,h,w
		for i in range(2*self.maxdisp+1):
			ind = i-self.maxdisp
			for j in range(2*int(self.maxdisp//1)+1):
				indd = j-int(self.maxdisp//1)
				feata = refimg_fea[:,:,max(0,-indd):height-indd,max(0,-ind):width-ind]
				featb = targetimg_fea[:,:,max(0,+indd):height+indd,max(0,ind):width+ind]
				diff = (feata*featb)
				cost[:, :, i,j,max(0,-indd):height-indd,max(0,-ind):width-ind]   = diff
		cost = F.leaky_relu(cost, 0.1,inplace=True)
		return cost