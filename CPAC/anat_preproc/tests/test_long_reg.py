

def test_long_reg_anat():
    """
    Test of the template creation from within the cpac docker image
    Paths are hardcoded for nuw, until we have a real test dataset
    Returns
    -------

    """
    import os
    import yaml
    import CPAC.anat_preproc.longitudinal_workflow as lw
    # import CPAC.anat_preproc.longitudinal_preproc as lp
    from CPAC.utils import Configuration
    data_config_file = '/outputs/cpac_data_config_custom_sub-0027225.yml'
    output_folder = '/outputs/test_long_reg'
    pipeline_config_file = '/outputs/cpac_pipeline_config_custom_long_reg.yml'

    if not os.path.exists(pipeline_config_file):
        raise IOError("config file %s doesn't exist" % pipeline_config_file)
    else:
        c = Configuration(yaml.load(open(pipeline_config_file, 'r')))
        print("DEBUG: import pipeline config")

    try:
        with open(data_config_file, 'r') as sf:
            sublist = yaml.load(sf)
            print("DEBUG: import data config")
    except IOError:
        print("Subject list is not in proper YAML format. Please check "
              "your file")
        raise Exception

    if hasattr(c, 'gen_custom_template') and c.gen_custom_template:
        subject_id_dict = {}
        for sub in sublist:
            if sub['subject_id'] in subject_id_dict:
                subject_id_dict[sub['subject_id']].append(sub)
            else:
                subject_id_dict[sub['subject_id']] = [sub]
        print("DEBUG: formatting subject dictionary")
        # subject_id_dict has the subject_id as keys and a list of sessions for
        # each participant as value
        # TODO modify it so the functions can be called in separated nodes
        lw.anat_longitudinal_workflow(subject_id_dict, c)
