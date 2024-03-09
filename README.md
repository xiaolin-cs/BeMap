# BeMap
<h3 align="center">BeMap: Balanced Message Passing for Fair Graph Neural Network (LOG 2023) </h3>
<h3 align="center"> 
  Links: 
  <a href="https://arxiv.org/pdf/2306.04107.pdf"> Paper </a> | 
  <a href="https://openreview.net/attachment?id=4RiLDrCbzW&name=poster_preview"> Poster </a> |  
  <a href="https://arxiv.org/abs/2306.04107"> arXiv </a>
</h3>

BeMap is a *fair message passing* method for solving the unfairness problem on graph machine learning. It is a model-agnostic easy-to-implement solution that is able to effectively mitigate *structural bias* for most typical GNNs, such as GCN and GAT.

## Cite Us
If you find this repository helpful in your work or research, we would greatly appreciate citations to the following paper:

```bibtex
@inproceedings{lin2023bemap,
  title={BeMap: Balanced Message Passing for Fair Graph Neural Network},
  author={Lin, Xiao and Kang, Jian and Cong, Weilin and Tong, Hanghang},
  booktitle={The Second Learning on Graphs Conference},
  year={2023}
}
```

## Background
We introduce a novel fair message passing method named BeMap. It creates a fair graph structure for each epoch by leveraging a balance-aware sampling strategy to balance the number of the 1-hop neighbors of each node among different demographic groups. The figure below gives an overview of the MESA framework.

