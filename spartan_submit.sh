#!/bin/bash

# Iterate through the command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --conda-env)
            conda_env="$2"
            shift
            shift
            ;;
        --script-name)
            script_name="$2"
            shift
            shift
            ;;
        --arguments)
            arguments="$2"
            shift
            shift
            ;;
        --runtime)
            runtime="$2"
            shift
            shift
            ;;
        --mem)
            mem="$2"
            shift
            shift
            ;;
        --cpus-per-task)
            cpus_per_task="$2"
            shift
            shift
            ;;
        --a100-gpus)
            a100_gpus="$2"
            shift
            shift
            ;;
        --deeplearn-gpus)
            deeplearn_gpus="$2"
            shift
            shift
            ;;
        --id)
            id="$2"
            shift
            shift
            ;;
        *)
            # Skip unknown arguments
            shift
            ;;
    esac
done

# Set the script and screenshots
timeStamp=$(date +"%Y%m%d%H%M%S")
script_name_ss="$timeStamp-$id-$script_name"
cp -- "$script_name" "slurm/$script_name_ss"

# cp -- "tracker.py" "slurm/tracker.py"

run_command="python slurm/$script_name_ss $arguments"

# task_filenames="a100.$script_name_ss.slurm"
# task_filenames="deeplearn.$script_name_ss.slurm"
# task_filenames="feit.a100.$script_name_ss.slurm"
task_filenames="a100.$script_name_ss.slurm feit.a100.$script_name_ss.slurm deeplearn.$script_name_ss.slurm"

mkdir -p slurm

# Deeplearn 
echo "#!/bin/bash
#SBATCH --job-name=deeplearn.$script_name_ss
#SBATCH --output=slurm/deeplearn."$script_name_ss".log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --time=$runtime
#SBATCH --mem=$mem
#SBATCH --partition=deeplearn
#SBATCH --gres=gpu:$deeplearn_gpus
#SBATCH -q gpgpudeeplearn
#SBATCH -A punim1629
#SBATCH --mail-user=s.liu96@student.unimelb.edu.au
#SBATCH --mail-type=ALL

echo \"Load module...\"
module load GCCcore/11.3.0
module load Xvfb/1.20.13
module load X11/20220504
module load FFmpeg/4.4.2

module load Anaconda3/2022.10
export CONDA_ENVS_PATH=/data/gpfs/projects/punim1629/anaconda3/envs

echo \"Activate conda env...\"
eval \"\$(conda shell.bash hook)\"
conda activate $conda_env

echo \"Good to go!\"
$run_command" > "slurm/deeplearn.$script_name_ss.slurm"

# A100
echo "#!/bin/bash
#SBATCH --job-name=a100.$script_name_ss
#SBATCH --output=slurm/a100."$script_name_ss".log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --time=$runtime
#SBATCH --mem=$mem
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:$a100_gpus
#SBATCH -A punim1629
#SBATCH --mail-user=s.liu96@student.unimelb.edu.au
#SBATCH --mail-type=ALL

echo \"Load module...\"
module load GCCcore/11.3.0
module load Xvfb/1.20.13
module load X11/20220504
module load FFmpeg/4.4.2

module load Anaconda3/2022.10
export CONDA_ENVS_PATH=/data/gpfs/projects/punim1629/anaconda3/envs

echo \"Activate conda env...\"
eval \"\$(conda shell.bash hook)\"
conda activate $conda_env

echo \"Good to go!\"
$run_command" > "slurm/a100.$script_name_ss.slurm"

# FEIT A100
echo "#!/bin/bash
#SBATCH --job-name=feit.a100.$script_name_ss
#SBATCH --output=slurm/feit.a100."$script_name_ss".log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --time=$runtime
#SBATCH --mem=$mem
#SBATCH --partition=feit-gpu-a100
#SBATCH --gres=gpu:$a100_gpus
#SBATCH -q feit
#SBATCH -A punim1629
#SBATCH --mail-user=s.liu96@student.unimelb.edu.au
#SBATCH --mail-type=ALL

echo \"Load module...\"
module load GCCcore/11.3.0
module load Xvfb/1.20.13
module load X11/20220504
module load FFmpeg/4.4.2

module load Anaconda3/2022.10
export CONDA_ENVS_PATH=/data/gpfs/projects/punim1629/anaconda3/envs

echo \"Activate conda env...\"
eval \"\$(conda shell.bash hook)\"
conda activate $conda_env

echo \"Good to go!\"
$run_command" > "slurm/feit.a100.$script_name_ss.slurm"

for task_filename in $task_filenames
do
    sbatch slurm/$task_filename
done

