**Notebooks**

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
