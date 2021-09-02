#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self,e_word):
        """ initial two Linear to construct highway connection
        @param  (int):the dimention of the word

        """
        super(Highway,self).__init__()
        self.proj = nn.Linear(e_word, e_word)
        self.gate = nn.Linear(e_word, e_word)
        
    
    def forward(self,conv_out:torch.Tensor) -> torch.Tensor:
        """ highway connection
        @param conv_out (torch.Tensor): the convolution layer's output,its shape is (batch_size, e_word)
        @return highway_out (torch.Tensor): tensor of (batch_size, e_word)
        """
        
        x_proj = F.relu(self.proj(conv_out))
        x_gate = torch.sigmoid(self.gate(conv_out))
        highway_out = torch.mul(x_proj,x_gate) + torch.mul(conv_out,1-x_gate) #点乘
        
        return highway_out        
### END YOUR CODE 

if __name__ == '__main__':
    high = Highway(5)
    input = torch.randn(4,5)
    pred = high(input)
    assert(pred.shape==input.shape)