from pydantic import BaseModel

class FileOutputPaths(BaseModel):
    vocab: str = "local/vocab_shared.pkl"


class Constants(BaseModel):
    file_output_paths: FileOutputPaths = FileOutputPaths()

constants = Constants()