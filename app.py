from flask import Flask, render_template, request
import torch, torchtext, nltk, math
from torch import nn
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split

app = Flask(__name__)

with open('sherlockholmes.txt', 'r', encoding='utf-8') as file:
    sherlock_data = file.read()
nltk.download('punkt')

sherlock_corpus = sent_tokenize(sherlock_data)
# Assuming 'sentences' is the list of sentences you obtained from the previous step

# Split into training and temporary set (temp_set)
temp_set, test_set = train_test_split(sherlock_corpus, test_size=0.2, random_state=43)

# Split temp_set into training and validation sets
train_set, val_set = train_test_split(temp_set, test_size=0.25, random_state=43)

train_set_token = [word_tokenize(sentence) for sentence in train_set]

vocab = torchtext.vocab.build_vocab_from_iterator(train_set_token, min_freq=3)
vocab.insert_token('<unk>', 0)
vocab.insert_token('<eos>', 1)
vocab.set_default_index(vocab['<unk>'])

 

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim    = hid_dim
        self.emb_dim    = emb_dim
        
        self.embedding  = nn.Embedding(vocab_size, emb_dim)
        self.lstm       = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc         = nn.Linear(hid_dim, vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hid_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_other)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.emb_dim,
                self.hid_dim).uniform_(-init_range_other, init_range_other) #We
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hid_dim,   
                self.hid_dim).uniform_(-init_range_other, init_range_other) #Wh
    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell   = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell
        
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach() #not to be used for gradient computation
        cell   = cell.detach()
        return hidden, cell
        
    def forward(self, src, hidden):
        #src: [batch_size, seq len]
        embedding = self.dropout(self.embedding(src)) #harry potter is
        #embedding: [batch-size, seq len, emb dim]
        output, hidden = self.lstm(embedding, hidden)
        #ouput: [batch size, seq len, hid dim]
        #hidden: [num_layers * direction, seq len, hid_dim]
        output = self.dropout(output)
        prediction =self.fc(output)
        #prediction: [batch_size, seq_len, vocab_size]
        return prediction, hidden



def generate(prompt, max_seq_len, temperature, model, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = word_tokenize(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            #prediction: [batch size, seq len, vocab size]
            #prediction[:, -1]: [batch size, vocab size] #probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction) #autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens

             

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        user_input = request.form['user_input']
        vocab_size = len(vocab)
        emb_dim = 1024                # 400 in the paper
        hid_dim = 1024                # 1150 in the paper
        num_layers = 2                # 3 in the paper
        dropout_rate = 0.65              
        lr = 1e-3       
        
        model_path = 'best-val-lstm_lm.pt'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers,dropout_rate)  
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        temperatures = [1.0]
        for temperature in temperatures:
            generation = generate(user_input, 30, temperature, model, 
                          vocab, device, 0)
            
        result = '\n'+' '.join(generation)+'\n'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
