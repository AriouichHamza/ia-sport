
import torch





#save model
def save_model(model, N_EPOCHS, optimizer, PATH, loss=0.4):
    torch.save({
        'epoch': N_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, PATH)

#load saved model
def load_model(model, PATH, optimizer):
    #print("loading model: ", PATH)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("Loaded model : epoch {} - Loss {}: ".format(epoch,loss))
    #model.eval()
    # - or -
    # model.train()