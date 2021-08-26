#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self,e_char,filter_size,m_word = 21,kernel_size = 5):
        """initial cnn
        @param filter_size (int): the number of filter,also the e_word
        @param e_char (int): the demention of char
        @param kernel_size (int): the filter's length
        """        
        super(CNN,self).__init__()
        self.conv1d = nn.Conv1d(e_char,filter_size,kernel_size)
        self.pool = nn.MaxPool1d(m_word - kernel_size + 1)
        
    def forward(self,reshaped) -> torch.Tensor:
        """
        @param reshaped (torch.Tensor): the char embedding of sentences, which is tensor of (batch_size,e_char,max_word_len)
        @return conv_out (torch.Tensor):the ouput of cnn, which is tensor of (bat_size,e_word)
        """
        conv_out = F.relu(self.conv1d(reshaped))
        conv_out = self.pool(conv_out)
        
        return conv_out.squeeze(-1) #最后一个维度被maxpool了，因此将最后一个维度去掉
        
        
### END YOUR CODE

if __name__ == '__main__':
    cnn = CNN(50,4) #(char_embedding, filter_number or word_embedding)
    input = torch.randn(10,50,21) #(batch_size or words, char_embedding, max_word_len)
    assert(cnn(input).shape==(10,4))
 