# Code for ``Interpretable EEG biomarkers for neurological disease models in mice using bag-of-waves classifiers''

Link to the [paper](http://doi.org/10.1088/1741-2552/ae4d8c).

The raw data is available at the following [DOI](https://zenodo.org/records/18577633).


Intermediate data products are not currently available (3/10/2026) as they were stored as pickle files. More safe file formats will be included in the future.

## Steps
- Download the raw EEG data (EDF files) from Zenodo: [DOI](https://zenodo.org/records/18577633)
- The meta-data file ([Jax_mice_with_splits_df.csv](https://github.com/ajbrockmeier/bowaves-mice-genotyping/blob/main/meta-data/Jax_mice_with_splits_df.csv)) contains the list of paths to file names, seizure exclusion zones, chronological splits, train-test split assignments to form a dataframe. By default the EDF files are assumed to be stored in '/Users/Shared/Mice_Jax_Lab', but the paths can be edited. Run make_metadata_frame.py to produce store the pandas dataframe in pickle format that will be used by other functions.   
- Dictionary learning (shift-invariant k-means with cosine similarity) is conducted by get_dict_by_split.py
- Counts of waveforms (or spectra clusters) are extracted by get_counts_by_split.py
- Classifiers are fit by get_classifiers_by_split.py
- Baseline methods involving [Hydra](https://github.com/angus924/hydra/blob/main/code/hydra.py) and softcount features are GNU GPL licensed and are available by request (email: ajbrock (at) udel.edu) and require installing torch (not included in the requirements.txt)
  

**Notebooks**
(The first notebook implements the SHAP value analysis. The last notebook is run after the others to produce the ROC curves. The notebooks produce the figures used in the paper as well as the tabular results)

cmd_analyze_shapely_LOO_pooled.ipynb

cmd_analyze_classifier_LOO_spectral.ipynb

cmd_analyze_classifier_LOO_pooled.ipynb

make_ROC_curves.ipynb 


**Example script calls** 

~~~
python get_dict_by_split.py --fold $fold --split $split --class1 $class1 --class2 $class2  --windows 40000 
~~~

The variables above are taken from the columns in ``dict_job_array.csv``


~~~
python get_counts_by_split.py --fold $fold --split $split --learn $learn --class1 $class1  --class2 $class2  --segments 480 --windows 40000 --hours_in_segment 1
~~~

The variables are taken from the columns in ``count_jobs.csv``

~~~
python get_classifiers_by_split.py --fold $fold --split $split --task $task --windows 40000 --loo
~~~

The variables are taken from the columns in ``classifier_job_args.csv``
