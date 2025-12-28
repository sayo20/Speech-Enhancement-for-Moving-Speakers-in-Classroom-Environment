#!/bin/bash
#SBATCH --job-name="mimo-sep-Adult-3e-4"       # Job name
#SBATCH --time=60:00                        # Wall time (HH:MM:SS)
#SBATCH --partition=gpu_a100                   # Partition name (e.g., GPU type)
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --ntasks-per-node=1                    # Tasks per node
#SBATCH --gpus=1                              # Number of GPUs (shorthand for --gres=gpu:1)
#SBATCH --cpus-per-task=8                     # Number of CPUs per task
#SBATCH --mail-type=END                        # When to send email (END, FAIL, etc.)
#SBATCH --mail-user=feyisayo.olalere@donders.ru.nl  # Email address

# Load environment modules
module purge
module load 2023
module load Anaconda3/2023.07-2

# Use EBROOTANACONDA3 to correctly source conda.sh
# source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda init bash
source ~/.bashrc
conda activate audioSpices

# Navigate to your script directory (use quotes to handle spaces!)
# cd "$HOME/Year 4/ClassroomSeparation"
echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Run the Python script
python Test_localizer.py
# python Train_localizer.py, Train_TC_quant Test_localizer
# python Train_enhancement_TC.py,Train_TC_finetune
# python Test_TC.py, Test_enhancement.py, Test_TC_finetune