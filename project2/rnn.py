from torch import nn


class Model(nn.Module):

    def __init__(self, vocabulary_size, embed_size, hidden_size, output_size,batch_size):
        super(Model, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size


        self.embed = nn.Embedding(vocabulary_size,embed_size)
        self.lstm = nn.LSTM(input_size = input_size,hidden_size = hidden_size, batch_first = true)
        self.output = nn.Linear(hidden_size,output_size)
    
    def forward(self,x,hidden):
            x = x.view(batch_size, sequence_length, input_size)
            out, hidden = self.rnn(x,hidden)
            out = out.view(-1,num_classes)
            return hidden, out


    def init_hidden(self):
        return Variable(torch.zeros(num_layers, batch_size, hidden_size))