import subprocess
import os

class CascadeXMLWrapper:
    """
    A high-level API for CascadeXML, a multi-stage XMLC architecture 
    that balances prediction accuracy and inference throughput.
    """
    def __init__(self, root_path, src_path="models/CascadeXML/src"):
        """
        Initializes the wrapper with project paths.
        """
        self.root_path = root_path
        self.main_script = os.path.join(root_path, src_path, "main.py")

    def train(self, dataset_path, num_labels, epochs=10, batch_size=32, lr=5e-5):
        """
        Triggers the training process using BERT-base as the encoder.
        
        Args:
            dataset_path (str): Path to the preprocessed dataset.
            num_labels (int): Total number of labels in the vocabulary.
            epochs (int): Number of training iterations.
        """
        cmd = [
            "python", self.main_script,
            "--dataset", dataset_path,
            "--num_labels", str(num_labels),
            "--num_epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", f"{lr:.1e}",
            "--bert", "bert-base-uncased",
            "--eval_scheme", "weighted"
        ]
        print(f"Starting CascadeXML training for {num_labels} labels...")
        subprocess.run(cmd, check=True)

    def predict(self, dataset_path, model_name, save_dir):
        """
        Executes inference to generate prediction scores in .npz format.
        """
        print(f"Running inference for {model_name}...")
        # Add logic to call main.py with --mode predict or similar flags