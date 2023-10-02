import time
import math
import torch

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder((target_tensor, encoder_hidden, encoder_outputs))

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.reshape(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def val_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):
    with torch.no_grad():
        total_loss = 0
        for data in dataloader:
            input_tensor, target_tensor = data
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder((target_tensor, encoder_hidden, encoder_outputs))
            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.reshape(-1)
            )

            total_loss += loss.item()
            pred_indices = torch.argmax(decoder_outputs, dim=-1)
            accuracy = torch.all(target_tensor == pred_indices, dim=1).sum()/len(target_tensor)

        return total_loss / len(dataloader), accuracy
    
