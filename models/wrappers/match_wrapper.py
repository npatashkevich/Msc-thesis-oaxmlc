import subprocess
import os

class MATCHWrapper:
    """
    A high-level API for the MATCH (Metadata-Aware Extreme Classification) model.
    MATCH leverages label hierarchies and document metadata to improve 
    classification performance on large-scale datasets like OpenAlex.
    """

    def __init__(self, root_path, model_path="models/MATCH"):
        """
        Initializes the wrapper with project paths.

        Args:
            root_path (str): Absolute path to the repository root.
            model_path (str): Relative path to the MATCH source code.
        """
        self.root_path = root_path
        self.main_script = os.path.join(root_path, model_path, "main.py")
        self.eval_script = os.path.join(root_path, model_path, "evaluation.py")

    def train(self, dataset_config, model_config):
        """
        Runs the training phase of the MATCH model.
        
        MATCH uses metadata-aware encoders and a hierarchical label structure 
        to handle the extreme label space efficiently.
        """
        cmd = [
            "python", self.main_script,
            "--data-cnf", dataset_config,
            "--model-cnf", model_config,
            "--mode", "train"
        ]
        print(f"Status: Initiating MATCH training session.")
        print(f"Config: {dataset_config}")
        subprocess.run(cmd, check=True)

    def predict(self, dataset_config, model_config, checkpoint=None):
        """
        Generates predictions and scores using a trained checkpoint.
        
        Args:
            dataset_config (str): Path to dataset YAML configuration.
            model_config (str): Path to model YAML configuration.
            checkpoint (str, optional): Path to the .pt checkpoint file.
        """
        cmd = [
            "python", self.main_script,
            "--data-cnf", dataset_config,
            "--model-cnf", model_config,
            "--mode", "eval"
        ]
        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])
            
        print(f"Status: Running inference to generate label scores.")
        subprocess.run(cmd, check=True)

    def evaluate(self, results_npy, scores_npy, targets_npy, train_labels_npy):
        """
        Executes the specialized evaluation script for MATCH.
        Calculates Ranking (P@k, nDCG), Coverage (BN-Coverage), 
        and Propensity-Scored metrics.
        """
        cmd = [
            "python", self.eval_script,
            "--results", results_npy,
            "--scores", scores_npy,
            "--targets", targets_npy,
            "--train-labels", train_labels_npy
        ]
        print("Status: Calculating final metrics (Ranking, Coverage, Propensity)...")
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # Example setup for the OpenAlex-1M experiment
    BASE_DIR = "/path/to/MSc-thesis-openalex-xmlc"
    match_model = MATCHWrapper(root_path=BASE_DIR)
    
    # Configuration paths
    DATA_CONF = "configure/datasets/OpenAlex-1M.yaml"
    MODEL_CONF = "configure/MATCH-OpenAlex.yaml"
    
    print("MATCH Wrapper successfully initialized.")
    # To start training:
    # match_model.train(DATA_CONF, MODEL_CONF)