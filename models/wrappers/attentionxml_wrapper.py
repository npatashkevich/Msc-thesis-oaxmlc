import subprocess
import os

class AttentionXMLWrapper:
    """
    A professional wrapper for the AttentionXML model to streamline 
    the training, prediction, and evaluation workflows.
    """

    def __init__(self, root_path, model_path="models/AttentionXML"):
        """
        Initializes the wrapper with the project root and model directory.

        Args:
            root_path (str): The absolute path to the project root.
            model_path (str): Relative path to the AttentionXML implementation.
        """
        self.root_path = root_path
        self.main_script = os.path.join(root_path, model_path, "main.py")
        self.eval_script = os.path.join(root_path, model_path, "evaluation.py")

    def train(self, dataset_config, model_config):
        """
        Triggers the training phase of the model.

        Args:
            dataset_config (str): Path to the dataset configuration file (.yaml).
            model_config (str): Path to the model hyperparameters file (.yaml).
        """
        cmd = [
            "python", self.main_script,
            "--data-cnf", dataset_config,
            "--model-cnf", model_config,
            "--mode", "train"
        ]
        print(f"Status: Starting training session using {dataset_config}")
        subprocess.run(cmd, check=True)

    def predict(self, dataset_config, model_config, checkpoint):
        """
        Runs the model in evaluation mode to generate label predictions and scores.

        Args:
            dataset_config (str): Path to the dataset configuration file.
            model_config (str): Path to the model configuration file.
            checkpoint (str): Path to the trained model checkpoint (.pt).
        """
        cmd = [
            "python", self.main_script,
            "--data-cnf", dataset_config,
            "--model-cnf", model_config,
            "--mode", "eval",
            "--checkpoint", checkpoint
        ]
        print(f"Status: Generating predictions using checkpoint: {checkpoint}")
        subprocess.run(cmd, check=True)

    def run_full_evaluation(self, pred_labels, pred_scores, targets, train_labels, binarizer, threshold=0.360):
        """
        Executes the evaluation script to calculate ranking and classification metrics.

        Args:
            pred_labels (str): Path to the predicted labels file (.npy).
            pred_scores (str): Path to the predicted scores file (.npy).
            targets (str): Path to the ground-truth test labels.
            train_labels (str): Path to training labels (used for propensity scores).
            binarizer (str): Path to the label binarizer object (.pkl).
            threshold (float): Classification threshold for F1 metrics.
        """
        cmd = [
            "python", self.eval_script,
            "--pred-labels", pred_labels,
            "--pred-scores", pred_scores,
            "--targets", targets,
            "--train-labels", train_labels,
            "--labels-binarizer", binarizer,
            "--cls-threshold", str(threshold)
        ]
        print("Status: Calculating P@k, nDCG, and Propensity-Scored metrics...")
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # Standard setup for the OpenAlex Extreme Multi-Label Classification task
    BASE_DIR = "/path/to/your/project"
    at_xml = AttentionXMLWrapper(root_path=BASE_DIR)
    
    # Example paths for the 1M document subset
    DATA_CONF = "configure/datasets/OpenAlex-1M.yaml"
    MODEL_CONF = "configure/AttentionXML-OpenAlex.yaml"
    
    print("AttentionXML Wrapper successfully initialized.")
    # To start training, uncomment the line below:
    # at_xml.train(DATA_CONF, MODEL_CONF)