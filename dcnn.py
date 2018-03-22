import dataloader.sst

if __name__ == '__main__':
    train_iter, val_iter, test_iter = dataloader.sst.load()
    for batch in train_iter:
        print(batch.text, batch.label)
