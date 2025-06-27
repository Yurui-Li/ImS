import torch.nn as nn
import torch.nn.functional as F
import torch

class ImSAgent(nn.Module):
    """
    MSP agent without bottle layer sharing
    """
    def __init__(self, input_shape, args):
        super(ImSAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_experts = args.n_experts

        self.bottoms = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(input_shape, args.rnn_hidden_dim),
                nn.ReLU())
             for _ in range(args.n_experts)])

        self.rnns = nn.ModuleList(
            [nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
             for _ in range(args.n_experts)])

        self.tops = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.rnn_hidden_dim, args.n_actions))
             for _ in range(args.n_experts)])

        self.gate = nn.Linear(input_shape, args.n_experts)

    def init_hidden(self,i):
        # make hidden states on same device as model
        return self.bottoms[i][0].weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()
        inputs = inputs.view(-1, e)
        #inputs shape:[b*a,input_shape]
        # expert calculate
        x = [bot(inputs) for bot in self.bottoms] #[b*a,rnn_hidden_dim]
        h_in = [h.reshape(-1, self.args.rnn_hidden_dim) for h in hidden_state]
        h_out= [self.rnns[i](x[i],h_in[i]) for i in range(self.n_experts)]
        experts_out = torch.stack([self.tops[i](h_out[i]) for i in range(self.n_experts)],dim=0) #[n_experts,b*a,n_actions]
        
        weight = F.sigmoid(self.gate(inputs)) #[b*a,n_experts]     
        weight = weight.T.unsqueeze(-1) #[n_experts,b*a,1]
        weight = weight.expand(experts_out.size()) #[n_experts,b*a,n_actions]

        # merge
        final_q = (experts_out * weight).mean(dim=0) #[b*a,n_actions]     

        return final_q.view(b, a, -1), h_out, experts_out.transpose(0,1)


