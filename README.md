# CHI-FEXP

This is a simple wrapper around pyradiomics to allow parallel feature extraction, flexible specification of label numbers in the data set file, and to facilitate running subsets of the data, allowing batching of feature extraction across multiple jobs.

## Installation
As of 2025/07/11, the version of `pyradiomics` on pyPI won't build successfully when installed via `pip install pyradiomics` in newer versions of python. In my experience on Windows 10, and `python 3.13`, you can install pyradiomics, and this package as follows:

```bash
# Make a virtual environment
python -m venv <venv-path>

# Activate the venv
<venv-path>/Scripts/activate # windows, OR
source <venv-path>/bin/activate # Linux

# Install pyradiomics
git clone git@github.com:AIM-Harvard/pyradiomics.git
python -m pip install ./pyradiomics

# Install chi-fexp
git clone git@github.com:jpeoples/chi-fexp.git
python -m pip install chi-fexp.git
```

## Getting started with `chi-fexp`

### A simple example of a very basic feature extraction

The data for this example is laid out as follows:

```
<ROOT>/
  <Subject_ID>/
    <Subject_ID>_volume.nii.gz # This is the CT scan
    <Subject_ID>_Tumor.nii.gz  # This is a binary 
                               # mask with 0 as 
                               # background and 
                               # 1 indicating a tumor
```

A highly minimalistic data set file for this project would look like this (`dataset.csv`)

| Image                                     | Mask                                     |
| ----------------------------------------- | ---------------------------------------- |
| `<Subject_ID>/<Subject_ID>_volume.nii.gz` | `<Subject_ID>/<Subject_ID>_Tumor.nii.gz` |
| `<Subject_ID>/<Subject_ID>_volume.nii.gz` | `<Subject_ID>/<Subject_ID>_Tumor.nii.gz` |
| `<Subject_ID>/<Subject_ID>_volume.nii.gz` | `<Subject_ID>/<Subject_ID>_Tumor.nii.gz` |
| `<Subject_ID>/<Subject_ID>_volume.nii.gz` | `<Subject_ID>/<Subject_ID>_Tumor.nii.gz` |
| ...                                       | ...                                      |

Notice that `<ROOT>` is not specified in the data set file. Instead, the paths are specified relative to `<ROOT>`. The `<ROOT>` directory is then specified separately on the command line, making it easy to relocate data sets and reuse the data set file.

Suppose we have a pyradiomics config file called `config.yaml`. For more information on `pyradiomics` config files see the examples [here](https://github.com/AIM-Harvard/pyradiomics/tree/master/examples/exampleSettings), and the docs [here](https://pyradiomics.readthedocs.io/en/latest/customization.html#parameter-file)

A few sample configurations are included in this repository as well: (here)[/chi/fexp/configs/].

To extract features:

```bash
python -m chi.fexp --conf config.yaml \
                   --output feature_table.csv \
                   --dataset dataset.csv \
                   --dataset_root <ROOT> \
                   --jobs -1
```

Breaking down each argument: we tell the program to use `config.yaml`, output the features to `feature_table.csv`, use `dataset.csv` as the input dataset file, and find the actual imaging data in the root folder `<ROOT>`, and execute in parallel using as many jobs as there are cores (-1).

Since we don't specify a MaskLabel column in the data set file, the default label will be used (as set by the command line argument `--use_label`. Since we also don't specify this, the default value of 1 will be used.)

### A slightly more complex example

This time, suppose the file layout is the same, but the segmentation files contain an arbitrary number tumor segmentations, indicated by voxel values:

- 0: background
- 1: Tumor 1
- 2: Tumor 2
- ...
- N: Tumor N

Suppose further, that we are interested in extracting featuers from a single, specified tumor in each case. The required label varies across patients. This is easily accommodated using the MaskLabel column. Feature extraction can be done exactly as in the previous example, with an updated data set file:


| Image                                     | Mask                                     | MaskLabel |
| ----------------------------------------- | ---------------------------------------- | --------- |
| `<Subject_ID>/<Subject_ID>_volume.nii.gz` | `<Subject_ID>/<Subject_ID>_Tumor.nii.gz` | 1         |
| `<Subject_ID>/<Subject_ID>_volume.nii.gz` | `<Subject_ID>/<Subject_ID>_Tumor.nii.gz` | 4         |
| `<Subject_ID>/<Subject_ID>_volume.nii.gz` | `<Subject_ID>/<Subject_ID>_Tumor.nii.gz` | 2         |
| `<Subject_ID>/<Subject_ID>_volume.nii.gz` | `<Subject_ID>/<Subject_ID>_Tumor.nii.gz` | 3         |
| ...                                       | ...                                      | ...       |


### Combining labels
Now suppose we want to extract features from an ROI which is specified by combining multiple label values. In the MaskLabel column, multiple labels can be specified by separating the numbers by columns. Furthermore, `>` or `<` can be prepended to a number to get all label values greater or less than the specified value. For example:

- `MaskLabel=M,N` will combine labels M and N into a single mask
- `MaskLabel=>N` will use all labels greater than N
- `MaskLabel=<N` will use all labels less than N
- `MaskLabel=4,5,>7,<2` will combine labels 0,1,4,5, and 8,9,10... into a single mask

### Subsetting the data set file

Suppose I want to run the feature extraction in groups of `BATCH_SIZE`.

Then, I can use the `--start` and `--count` arguments to specify that the execution should begin at index `start` and run `count` rows.

To run the first batch:

```bash
python -m chi.fexp --conf config.yaml \
                   --output feature_table_0.csv \
                   --dataset dataset.csv \
                   --dataset_root <ROOT> \
                   --jobs -1 \
                   --start 0 \
                   --count $BATCH_SIZE
```

Then the next batches can be run in the same way, setting `start` to each multiple of `BATCH_SIZE` that is smaller than the total number of rows.

That is, for the Nth batch, we would run:

```bash
python -m chi.fexp --conf config.yaml \
                   --output feature_table_$N.csv \
                   --dataset dataset.csv \
                   --dataset_root <ROOT> \
                   --jobs -1 \
                   --start $((BATCH_SIZE * N)) \
                   --count $BATCH_SIZE \
```

**NB:** The output file must be different in invokation, because the output file is always overwritten.

**NB:** If `start + count` is larger than the total number of rows, the program will simply stop after the last row.

### Submitting a batched job to a SLURM cluster

This subsetting feature allows us to easily submit the feature extraction to a SLURM cluster, broken up into many small jobs. 

For example, suppose our data set file has 6006 rows. Now suppose we want to break this up into 601 small jobs to run on a SLURM cluster, where each job extracts features for 10 rows in the data set file (with the exception of the last job, which will run on 6 rows).

We can write the SLURM submission script as follows (`job.sh`)

```bash
#!/bin/bash
#
#SBATCH --array=0-600
#SBATCH --job-name=ExtractFeatures_%A_%a
#SBATCH -c 10
#SBATCH --mem 16G
#SBATCH -o slurm_logs/ExtractFeatures_%A_%a.out
#SBATCH -e slurm_logs/ExtractFeatures_%A_%a.err

# Load the correct version of python and venv
# Here I will assume just the system python is used.
# I also assume the venv is located in `.venv` in the
# working directory
source .venv/bin/activate

# Set the batch size, and the starting row index, based on the current array task ID
BATCH_SIZE=10
INDEX=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))

# Execute, with start and count set accordingly,
# outputting the feature table for the current batch to
# <pwd>/feature_batches/batch_<INDEX>.csv
# NOTE: The -u option passed to python requests python not
# to batch the output to stdout/stderr, ensuring that the
# output log files are updated more frequently during the
# slurm job's execution.

python -u -m chi.fexp --conf config.yaml \
                      --output feature_batches/batch_${INDEX}.csv \
                      --dataset dataset.csv \
                      --dataset_root <ROOT> \
                      --jobs $SLURM_CPUS_PER_TASK \
                      --start $INDEX \
                      --count $BATCH_SIZE \
```

Then this can be submitted to SLURM:

```bash
sbatch job.sh
```

The SBATCH commands at the top of the script have the following effects:

- `#SBATCH --array=0-600`: Submit jobs with Task IDs from 0 to 600, inclusive. The Task ID is available in the script as $SLURM_ARRAY_TASK_ID
- `#SBATCH --job-name=ExtractFeatures_%A_%a`: Give the job a name. %A gets the job number, while %a gives the current task ID in the array job
- `#SBATCH -c 10`: Request 10 cores for each job
- `#SBATCH --mem 16G`: Request 16 GB for each job
- `#SBATCH -o slurm_logs/ExtractFeatures_%A_%a.out`: Output stdout for the job to a file in slurm_logs/ directory, with the job number and task ID appended to the end
- `#SBATCH -e slurm_logs/ExtractFeatures_%A_%a.err`: Similar, for stderr.

## Command Line Usage

```
usage: python -m chi.fexp [-h] --conf CONF [CONF ...] --output OUTPUT [OUTPUT ...]
                          --dataset DATASET [--dataset_root DATASET_ROOT] [--jobs JOBS]        
                          [--image_column IMAGE_COLUMN] [--mask_column MASK_COLUMN]
                          [--label_column LABEL_COLUMN] [--dump_preprocessed]
                          [--dump_dir DUMP_DIR] [--use_label USE_LABEL] [--start START]        
                          [--count COUNT] [--resample_mask_before_extraction]

options:
  -h, --help            show this help message and exit
  --conf CONF [CONF ...]
                        A list of pyradiomics yaml configuration files (can be 1) (default:    
                        None)
  --output OUTPUT [OUTPUT ...]
                        A list of output feature csv files for each specified configuration    
                        (default: None)
  --dataset DATASET     A csv file specifying the images, masks, and labels for feature        
                        extraction (default: None)
  --dataset_root DATASET_ROOT
                        The root directory of the dataset, to which all paths in the data set  
                        file are relative (default: None)
  --jobs JOBS           The number of parallel jobs to use. Default is 1 (serial) (default:    
                        1)
  --image_column IMAGE_COLUMN
                        The name of the column specifying the image files. (default: Image)    
  --mask_column MASK_COLUMN
                        The name of the column specifying the mask file paths. (default:       
                        Mask)
  --label_column LABEL_COLUMN
                        The name of the column specifying the label(s) to use. (default:       
                        MaskLabel)
  --dump_preprocessed   Trigger the program to output preprocessed images, rather than
                        imaging features (default: False)
  --dump_dir DUMP_DIR   When outputting preprocessed images, this specifies the root path for  
                        output. (default: None)
  --use_label USE_LABEL
                        If no label_column is present in the data set, this argument
                        specifies the default label to use for all images (default: 1)
  --start START         Offset execution, processing the rows starting at the given index      
                        (default: -1)
  --count COUNT         Limit execution to this number of rows, starting from the start        
                        index. (default: -1)
  --resample_mask_before_extraction
                        This prevents certain rare errors.. (default: False)
```
