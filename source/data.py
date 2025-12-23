from src.data import TrainDataset, ValidDataset, create_dataloaders

def dataLoader(path, train_df, valid_df, img_size=720, pad_size=736):
    train_dataset = TrainDataset(path, train_df, img_size=img_size, pad_size=pad_size)
    valid_dataset = ValidDataset(path, valid_df, img_size=img_size, pad_size=pad_size)
    print(f'Train Images: {len(train_dataset)} Valid Images: {len(valid_dataset)}')
    return train_dataset, valid_dataset

