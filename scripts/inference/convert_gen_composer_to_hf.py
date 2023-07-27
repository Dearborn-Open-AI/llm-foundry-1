import sys
from composer.models import write_huggingface_pretrained_from_composer_checkpoint
import transformers

# Get the GCP paths from the command line arguments
composer_path = sys.argv[1]
hf_output_path = sys.argv[2]

# Convert the checkpoint
write_huggingface_pretrained_from_composer_checkpoint(
    composer_path, 
    hf_output_path
)

# Load the converted model
loaded_model = transformers.AutoModelForSequenceClassification.from_pretrained(hf_output_path)