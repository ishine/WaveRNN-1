import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import utils.logger as logger
import numpy as np
from layers.attention import EmbeddingAttention

class VectorQuant(nn.Module):
    """
        Input: (N, samples, n_channels, vec_len) numeric tensor
        Output: (N, samples, n_channels, vec_len) numeric tensor
    """
    def __init__(self, n_channels, n_classes, vec_len, num_group, num_sample, normalize=False):
        super().__init__()
        if normalize:
            target_scale = 0.06
            self.embedding_scale = target_scale
            self.normalize_scale = target_scale
        else:
            self.embedding_scale = 1e-3
            self.normalize_scale = None
        self.embedding0 = nn.Parameter(torch.randn(n_channels, n_classes, vec_len, requires_grad=True) * self.embedding_scale)
        self.offset = torch.arange(n_channels).cuda() * n_classes
        # self.offset: (n_channels) long tensor
        self.n_classes = n_classes
        self.after_update()

    def forward(self, x0):
        if self.normalize_scale:
            target_norm = self.normalize_scale * math.sqrt(x0.size(3))
            x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
            embedding = target_norm * self.embedding0 / self.embedding0.norm(dim=2, keepdim=True)
        else:
            x = x0
            embedding = self.embedding0
        #logger.log(f'std[x] = {x.std()}')
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        # x1: (N*samples, n_channels, 1, vec_len) numeric tensor

        # Perform chunking to avoid overflowing GPU RAM.
        index_chunks = []
        for x1_chunk in x1.split(512, dim=0):
            index_chunks.append((x1_chunk - embedding).norm(dim=3).argmin(dim=2))
        index = torch.cat(index_chunks, dim=0)
        # index: (N*samples, n_channels) long tensor
        if True: # compute the entropy
            hist = index.float().cpu().histc(bins=self.n_classes, min=-0.5, max=self.n_classes - 0.5)
            prob = hist.masked_select(hist > 0) / len(index)
            entropy = - (prob * prob.log()).sum().item()
            #logger.log(f'entrypy: {entropy:#.4}/{math.log(self.n_classes):#.4}')
        else:
            entropy = 0
        index1 = (index + self.offset).view(index.size(0) * index.size(1))
        # index1: (N*samples*n_channels) long tensor
        output_flat = embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1)
        # output_flat: (N*samples*n_channels, vec_len) numeric tensor
        output = output_flat.view(x.size())

        out0 = (output - x).detach() + x
        out1 = (x.detach() - output).float().norm(dim=3).pow(2)
        out2 = (x - output.detach()).float().norm(dim=3).pow(2) + (x - x0).float().norm(dim=3).pow(2)
        #logger.log(f'std[embedding0] = {self.embedding0.view(-1, embedding.size(2)).index_select(dim=0, index=index1).std()}')
        return (out0, out1, out2, entropy)

    def after_update(self):
        if self.normalize_scale:
            with torch.no_grad():
                target_norm = self.embedding_scale * math.sqrt(self.embedding0.size(2))
                self.embedding0.mul_(target_norm / self.embedding0.norm(dim=2, keepdim=True))


class VectorQuantGroup(nn.Module):
    """
        Input: (N, samples, n_channels, vec_len) numeric tensor
        Output: (N, samples, n_channels, vec_len) numeric tensor
    """
    def __init__(self, n_channels, n_classes, vec_len, num_group, num_sample, normalize=False):
        super().__init__()
        if normalize:
            target_scale = 0.06
            self.embedding_scale = target_scale
            self.normalize_scale = target_scale
        else:
            self.embedding_scale = 1e-3
            self.normalize_scale = None

        self.n_classes = n_classes
        self._num_group = num_group
        self._num_sample = num_sample
        if not self.n_classes % self._num_group == 0:
            raise ValueError('num of embeddings in each group should be an integer')
        self._num_classes_per_group = int(self.n_classes / self._num_group)

        self.embedding0 = nn.Parameter(torch.randn(n_channels, n_classes, vec_len, requires_grad=True) * self.embedding_scale)
        self.offset = torch.arange(n_channels).cuda() * n_classes
        # self.offset: (n_channels) long tensor
        self.after_update()

    def forward(self, x0):
        if self.normalize_scale:
            target_norm = self.normalize_scale * math.sqrt(x0.size(3))
            x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
            embedding = target_norm * self.embedding0 / self.embedding0.norm(dim=2, keepdim=True)
        else:
            x = x0
            embedding = self.embedding0
        #logger.log(f'std[x] = {x.std()}')
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        # x1: (N*samples, n_channels, 1, vec_len) numeric tensor

        # Perform chunking to avoid overflowing GPU RAM.
        index_chunks = []
        prob_chunks = []
        for x1_chunk in x1.split(512, dim=0):
            d = (x1_chunk - embedding).norm(dim=3)
			
			# Compute the group-wise distance
            d_group = torch.zeros(x1_chunk.shape[0], 1, self._num_group).to(torch.device('cuda'))
            for i in range(self._num_group):
                d_group[:, :, i] = torch.mean(
                    d[:, :, i * self._num_classes_per_group: (i + 1) * self._num_classes_per_group], 2)
					
			# Find the nearest group
            index_chunk_group = d_group.argmin(dim=2)

            # Generate mask for the nearest group
            index_chunk_group = index_chunk_group.repeat(1, self._num_classes_per_group)
            index_chunk_group = torch.mul(self._num_classes_per_group, index_chunk_group)
            idx_mtx = torch.LongTensor([x for x in range(self._num_classes_per_group)]).unsqueeze(0).cuda()
            index_chunk_group += idx_mtx
            encoding_mask = torch.zeros(x1_chunk.shape[0], self.n_classes).cuda()
            encoding_mask.scatter_(1, index_chunk_group, 1)
			
			# Compute the weight atoms in the group
            encoding_prob = torch.div(1, d.squeeze())
			
			# Apply the mask
            masked_encoding_prob = torch.mul(encoding_mask, encoding_prob)
            p, idx = masked_encoding_prob.sort(dim=1, descending=True)
            prob_chunks.append(p[:, :self._num_sample])
            index_chunks.append(idx[:, :self._num_sample])



        index = torch.cat(index_chunks, dim=0)
        prob_dist = torch.cat(prob_chunks, dim=0)
        prob_dist = F.normalize(prob_dist, p=1, dim=1)
        # index: (N*samples, n_channels) long tensor
        if True: # compute the entropy
            hist = index[:, 0].float().cpu().histc(bins=self.n_classes, min=-0.5, max=self.n_classes - 0.5)
            prob = hist.masked_select(hist > 0) / len(index)
            entropy = - (prob * prob.log()).sum().item()
            #logger.log(f'entrypy: {entropy:#.4}/{math.log(self.n_classes):#.4}')
        else:
            entropy = 0
        index1 = (index + self.offset)
        # index1: (N*samples*n_channels) long tensor
        output_list = []
        for i in range(self._num_sample):
            output_list.append(torch.mul(embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1[:, i]), prob_dist[:, i].unsqueeze(1).detach()))

        output_cat = torch.stack(output_list, dim=2)
        output_flat = torch.sum(output_cat, dim=2)
        # output_flat: (N*samples*n_channels, vec_len) numeric tensor
        output = output_flat.view(x.size())

        out0 = (output - x).detach() + x
        out1 = (x.detach() - output).float().norm(dim=3).pow(2)
        out2 = (x - output.detach()).float().norm(dim=3).pow(2) + (x - x0).float().norm(dim=3).pow(2)
        #logger.log(f'std[embedding0] = {self.embedding0.view(-1, embedding.size(2)).index_select(dim=0, index=index1).std()}')
        return (out0, out1, out2, entropy)

    def after_update(self):
        if self.normalize_scale:
            with torch.no_grad():
                target_norm = self.embedding_scale * math.sqrt(self.embedding0.size(2))
                self.embedding0.mul_(target_norm / self.embedding0.norm(dim=2, keepdim=True))