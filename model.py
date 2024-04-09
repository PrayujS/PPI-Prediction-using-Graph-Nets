import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_adj
from torch.optim.lr_scheduler import MultiStepLR

class CustomGCNConv(MessagePassing):
  def __init__(self, in_channels, out_channels):
    super().__init__(aggr='add')
    self.lin = nn.Linear(in_channels, out_channels)

  def forward(self, x, edge_index, edge_attr):
    out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
    out = out.to(self.lin.weight.dtype)
    return self.lin(out)

  def message(self, x_j, edge_attr):
    return torch.cat([x_j, edge_attr], dim=1)



class GCNLayer(nn.Module):
  def __init__(self, num_features_pro, output_dim, num_layers, dropout):
    super(GCNLayer, self).__init__()


    self.protA_convs = nn.ModuleList([CustomGCNConv(num_features_pro, output_dim) for _ in range(num_layers)])
    self.protB_convs = nn.ModuleList([CustomGCNConv(num_features_pro, output_dim) for _ in range(num_layers)])

    self.dropout = nn.Dropout(dropout)
    self.relu = nn.LeakyReLU()

    self.fc = nn.Linear(output_dim, output_dim)

  def forward(self, protA_data, protB_data):

    x1, edge_index1, edge_attr1 = protA_data[0], protA_data[1], protA_data[2]
    for conv in self.protA_convs:
      x1 = conv(x1, edge_index1, edge_attr1)
      x1 = self.relu(x1)
      x1 = self.dropout(x1)


    x2, edge_index2, edge_attr2 = protB_data[0], protB_data[1], protB_data[2]
    for conv in self.protB_convs:
      x2 = conv(x2, edge_index2, edge_attr2)
      x2 = self.relu(x2)
      x2 = self.dropout(x2)
    x1 = self.fc(x1)
    x2 = self.fc(x2)

    return x1, x2

class GATLayer(nn.Module):
  def __init__(self, num_features_pro, output_dim, num_layers, dropout, num_heads):
    super(GATLayer, self).__init__()


    self.protA_gats = nn.ModuleList([GATConv(1, output_dim, heads=num_heads, dropout=dropout) for _ in range(num_layers)])
    self.protB_gats = nn.ModuleList([GATConv(1, output_dim, heads=num_heads, dropout=dropout) for _ in range(num_layers)])


    self.fc = nn.Linear(output_dim * num_heads, output_dim)

    self.dropout = nn.Dropout(dropout)
    self.relu = nn.LeakyReLU()

  def forward(self, protA_data, protB_data):

    x1, edge_index1, edge_attr1 = protA_data[0], protA_data[1], protA_data[2]
    for gat in self.protA_gats:
      x1 = gat(x1, edge_index1, edge_attr=edge_attr1)
      x1 = self.relu(x1)
      x1 = self.dropout(x1)


    x2, edge_index2, edge_attr2 = protB_data[0], protB_data[1], protB_data[2]
    for gat in self.protB_gats:
      x2 = gat(x2, edge_index2, edge_attr=edge_attr2)
      x2 = self.relu(x2)
      x2 = self.dropout(x2)

    x1 = self.fc(x1)
    x2 = self.fc(x2)

    return x1, x2

class AFF(nn.Module):
  def __init__(self, input_feature_dims):
    super(AFF, self).__init__()
    self.input_feature_dims = input_feature_dims

    # Attention mechanism
    self.attention = nn.Linear(input_feature_dims *2, 1)
    self.attention.weight = nn.Parameter(torch.Tensor(1, input_feature_dims *2))
    nn.init.xavier_uniform_(self.attention.weight)
    self.softmax = nn.Softmax()

    # Fusion layer
    self.fc = nn.Linear(input_feature_dims * 2, input_feature_dims)
    self.relu = nn.LeakyReLU(0.05)

  def forward(self, x1, x2):
    #print("Shape of x1:", x1.shape)
    #print("Shape of x2:", x2.shape)

    if x1.size(0) != x2.size(0):
      if x1.size(0) < x2.size(0):
        x1_padded = torch.nn.functional.pad(x1, (0,0,0, x2.size(0) - x1.size(0)), mode='constant', value=0)
        x_concat = torch.cat((x1_padded, x2), dim=1)
      else:
        x2_padded = torch.nn.functional.pad(x2, (0,0,0, x1.size(0) - x2.size(0)), mode='constant', value=0)
        x_concat = torch.cat((x1, x2_padded), dim=1)
    else:
      x_concat = torch.cat((x1, x2), dim=1)

    #print("Shape of x_concat:", x_concat.shape)


    attn_weights = self.softmax(self.attention(x_concat))
    #print("shape of attention weights:", attn_weights.shape)

    if x1.size(0) < x2.size(0):
      x1_padded = torch.nn.functional.pad(x1, (0,0,0, x2.size(0) - x1.size(0)), mode='constant', value=0)
      #print("shape of x1_padded:", x1_padded.shape)
      x1_weighted = torch.mul(x1_padded, attn_weights[:, :])
      x2_weighted = torch.mul(x2, attn_weights[:, :])
    else: #x1.size(0) > x2.size(0)
      x2_padded = torch.nn.functional.pad(x2, (0,0,0, x1.size(0) - x2.size(0)), mode='constant', value=0)
      #print("shape of x2_padded:", x2_padded.shape)
      x1_weighted = torch.mul(x1, attn_weights[:, :])
      x2_weighted = torch.mul(x2_padded, attn_weights[:, :])


    x_weighted_concat = torch.cat((x1_weighted, x2_weighted), dim=1)

    # Fusion layer
    fused_features = self.fc(x_weighted_concat)
    fused_features = self.relu(fused_features)
    #print('shape of final fused_features:', fused_features.shape)
    return fused_features

# Define MLP classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, classifier_hidden_dim, dropout=0.2):
        super(MLPClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, classifier_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(classifier_hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=0, keepdim=True)
        x = self.fc2(x)
        x = self.sigmoid(x)
        #print("shape of output:", x.shape)
        #print("output:", x)
        return x


class GCN_Model(nn.Module):
  def __init__(self, num_features_pro, classifier_hidden_dim, input_dim, num_layers, dropout):
    super(GCN_Model, self).__init__()
    self.gcn_layer = GCNLayer(num_features_pro, input_dim, num_layers, dropout)
    self.aff = AFF(input_dim)
    self.classifier = MLPClassifier(input_dim, classifier_hidden_dim)

  def forward(self, protA_data, protB_data):
    x1, x2 = self.gcn_layer(protA_data, protB_data)
    x = self.aff(x1, x2)
    output = self.classifier(x)
    return output

class GAT_Model(nn.Module):
  def __init__(self, num_features_pro, classifier_hidden_dim, input_dim, num_layers, dropout, num_heads):
    super(GAT_Model, self).__init__()
    self.gat_layer = GATLayer(num_features_pro, input_dim, num_layers, dropout, num_heads)
    self.aff = AFF(input_dim)
    self.classifier = MLPClassifier(input_dim, classifier_hidden_dim)

  def forward(self, protA_data, protB_data):
    x1, x2 = self.gat_layer(protA_data, protB_data)
    x = self.aff(x1, x2)
    output = self.classifier(x)
    return output