from pydantic import BaseModel

class Training(BaseModel):
    batch_size: int
    shuffle: bool
    learning_rate: float
    num_epochs: int
    device: str

class Model(BaseModel):
    num_tokens: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    dropout: float

class Data(BaseModel):
    train_split: float
    val_split: float
    random_seed: int
    dataset_path: str


class Config(BaseModel):
    training: Training
    model: Model
    data: Data
    logging_dir: str