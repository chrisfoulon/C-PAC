# -*- coding: utf-8 -*-
import os
import copy
import time
import shutil

from nipype import config
from nipype import logging
import nipype.pipeline.engine as pe
import nipype.interfaces.afni as afni

import CPAC
from CPAC.utils.datasource import (
    create_anat_datasource,
    create_func_datasource,
    create_check_for_s3_node
)

from CPAC.anat_preproc.anat_preproc import create_anat_preproc
from CPAC.func_preproc.func_preproc import (
    create_func_preproc,
    create_wf_edit_func
)
from CPAC.anat_preproc.longitudinal_preproc import template_creation_flirt

from CPAC.utils import Strategy, find_files, function

from CPAC.utils.utils import (
    create_log,
    check_config_resources,
    check_system_deps,
    get_scan_params,
    get_tr
)

logger = logging.getLogger('nipype.workflow')


def create_log_node(workflow, logged_wf, output, index, scan_id=None):
    try:
        log_dir = workflow.config['logging']['log_directory']
        if logged_wf:
            log_wf = create_log(wf_name='log_%s' % logged_wf.name)
            log_wf.inputs.inputspec.workflow = logged_wf.name
            log_wf.inputs.inputspec.index = index
            log_wf.inputs.inputspec.log_dir = log_dir
            workflow.connect(logged_wf, output, log_wf, 'inputspec.inputs')
        else:
            log_wf = create_log(wf_name='log_done_%s' % scan_id,
                                scan_id=scan_id)
            log_wf.base_dir = log_dir
            log_wf.inputs.inputspec.workflow = 'DONE'
            log_wf.inputs.inputspec.index = index
            log_wf.inputs.inputspec.log_dir = log_dir
            log_wf.inputs.inputspec.inputs = log_dir
            return log_wf
    except Exception as e:
        print(e)


def init_subject_wf(sub_dict, conf):
    c = copy.copy(conf)

    subject_id = sub_dict['subject_id']
    if sub_dict['unique_id']:
        subject_id += "_" + sub_dict['unique_id']

    log_dir = os.path.join(c.logDirectory, 'pipeline_%s' % c.pipelineName,
                           subject_id)
    if not os.path.exists(log_dir):
        os.makedirs(os.path.join(log_dir))

    config.update_config({
        'logging': {
            'log_directory': log_dir,
            'log_to_file': bool(getattr(c, 'run_logging', True))
        }
    })

    logging.update_logging(config)

    # Start timing here
    pipeline_start_time = time.time()
    # TODO LONG_REG change prep_worflow to use this attribute instead of the local var
    c.update('pipeline_start_time', pipeline_start_time)

    # Check pipeline config resources
    sub_mem_gb, num_cores_per_sub, num_ants_cores = \
        check_config_resources(c)

    # TODO LONG_REG understand and handle that
    # if plugin_args:
    #     plugin_args['memory_gb'] = sub_mem_gb
    #     plugin_args['n_procs'] = num_cores_per_sub
    # else:
    #     plugin_args = {'memory_gb': sub_mem_gb, 'n_procs': num_cores_per_sub}

    # perhaps in future allow user to set threads maximum
    # this is for centrality mostly
    # import mkl
    numThreads = '1'
    os.environ['OMP_NUM_THREADS'] = '1'  # str(num_cores_per_sub)
    os.environ['MKL_NUM_THREADS'] = '1'  # str(num_cores_per_sub)
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(num_ants_cores)

    # calculate maximum potential use of cores according to current pipeline
    # configuration
    max_core_usage = int(c.maxCoresPerParticipant) * int(
        c.numParticipantsAtOnce)

    information = """

        C-PAC version: {cpac_version}

        Setting maximum number of cores per participant to {cores}
        Setting number of participants at once to {participants}
        Setting OMP_NUM_THREADS to {threads}
        Setting MKL_NUM_THREADS to {threads}
        Setting ANTS/ITK thread usage to {ants_threads}
        Maximum potential number of cores that might be used during this run: {max_cores}

    """

    logger.info(information.format(
        cpac_version=CPAC.__version__,
        cores=c.maxCoresPerParticipant,
        participants=c.numParticipantsAtOnce,
        threads=numThreads,
        ants_threads=c.num_ants_threads,
        max_cores=max_core_usage
    ))

    # Check system dependencies
    check_system_deps(check_ants='ANTS' in c.regOption,
                      check_ica_aroma='1' in str(c.runICA[0]))

    # absolute paths of the dirs
    c.workingDirectory = os.path.abspath(c.workingDirectory)
    if 's3://' not in c.outputDirectory:
        c.outputDirectory = os.path.abspath(c.outputDirectory)

    # Workflow setup
    workflow_name = 'resting_preproc_' + str(subject_id)
    workflow = pe.Workflow(name=workflow_name)
    workflow.base_dir = c.workingDirectory
    workflow.config['execution'] = {
        'hash_method': 'timestamp',
        'crashdump_dir': os.path.abspath(c.crashLogDirectory)
    }

    # Extract credentials path if it exists
    try:
        creds_path = sub_dict['creds_path']
        if creds_path and 'none' not in creds_path.lower():
            if os.path.exists(creds_path):
                input_creds_path = os.path.abspath(creds_path)
            else:
                err_msg = 'Credentials path: "%s" for subject "%s" was not ' \
                          'found. Check this path and try again.' % (
                              creds_path, subject_id)
                raise Exception(err_msg)
        else:
            input_creds_path = None
    except KeyError:
        input_creds_path = None

    # TODO ASH normalize file paths with schema validator
    template_anat_keys = [
        ("anat", "template_brain_only_for_anat"),
        ("anat", "template_skull_for_anat"),
        ("anat", "ref_mask"),
        ("anat", "template_symmetric_brain_only"),
        ("anat", "template_symmetric_skull"),
        ("anat", "dilated_symmetric_brain_mask"),
        ("anat", "templateSpecificationFile"),
        ("anat", "lateral_ventricles_mask"),
        ("anat", "PRIORS_CSF"),
        ("anat", "PRIORS_GRAY"),
        ("anat", "PRIORS_WHITE"),
        ("other", "configFileTwomm"),
    ]

    for key_type, key in template_anat_keys:
        node = create_check_for_s3_node(
            key,
            getattr(c, key), key_type,
            input_creds_path, c.workingDirectory
        )

        setattr(c, key, node)

    if c.reGenerateOutputs is True:
        working_dir = os.path.join(c.workingDirectory, workflow_name)
        erasable = list(find_files(working_dir, '*sink*')) + \
                   list(find_files(working_dir, '*link*')) + \
                   list(find_files(working_dir, '*log*'))

        for f in erasable:
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f)

    return c, subject_id, input_creds_path


def func_template_generation(sub_list, conf):
    for func in sub_list:
        """
        truncate
        (Func_preproc){
        two step motion corr 
        refit 
        resample
        motion corr
        skullstripping
        mean + median
        }  
        dist corr and apply dist corr res
        config file registration target (epi t1)
        """
        if 'func' in func or 'rest' in func:
            if 'func' in sub_list:
                func_paths_dict = sub_list['func']
            else:
                func_paths_dict = sub_list['rest']

            subject_id = sub_list['subject_id']

            try:
                creds_path = sub_list['creds_path']
                if creds_path and 'none' not in creds_path.lower():
                    if os.path.exists(creds_path):
                        input_creds_path = os.path.abspath(creds_path)
                    else:
                        err_msg = 'Credentials path: "%s" for subject "%s" was not ' \
                                  'found. Check this path and try again.' % (
                                      creds_path, subject_id)
                        raise Exception(err_msg)
                else:
                    input_creds_path = None
            except KeyError:
                input_creds_path = None

            func_wf = create_func_datasource(func_paths_dict,
                                             'func_gather_%d' % str(subject_id))
            func_wf.inputs.inputnode.set(
                subject=subject_id,
                creds_path=input_creds_path,
                dl_dir=conf.workingDirectory
            )
            func_wf.get_node('inputnode').iterables = \
                ("scan", func_paths_dict.keys())

            # Add in nodes to get parameters from configuration file
            # a node which checks if scan_parameters are present for each scan
            scan_params = \
                pe.Node(
                    function.Function(input_names=['data_config_scan_params',
                                                   'subject_id',
                                                   'scan',
                                                   'pipeconfig_tr',
                                                   'pipeconfig_tpattern',
                                                   'pipeconfig_start_indx',
                                                   'pipeconfig_stop_indx'],
                                      output_names=['tr',
                                                    'tpattern',
                                                    'ref_slice',
                                                    'start_indx',
                                                    'stop_indx'],
                                      function=get_scan_params,
                                      as_module=True),
                    name='scan_params_%d' % str(subject_id))

            workflow_name = 'resting_preproc_' + str(subject_id)
            workflow = pe.Workflow(name=workflow_name)
            workflow.base_dir = conf.workingDirectory
            workflow.config['execution'] = {
                'hash_method': 'timestamp',
                'crashdump_dir': os.path.abspath(conf.crashLogDirectory)
            }

            if "Selected Functional Volume" in conf.func_reg_input:
                get_func_volume = pe.Node(interface=afni.Calc(),
                                          name='get_func_volume_%d' % str(
                                              subject_id))

                get_func_volume.inputs.set(
                    expr='a',
                    single_idx=conf.func_reg_input_volume,
                    outputtype='NIFTI_GZ'
                )
                workflow.connect(func_wf, 'outputspec.rest',
                                 get_func_volume, 'in_file_a')

            # wire in the scan parameter workflow
            workflow.connect(func_wf, 'outputspec.scan_params',
                             scan_params, 'data_config_scan_params')

            workflow.connect(func_wf, 'outputspec.subject',
                             scan_params, 'subject_id')

            workflow.connect(func_wf, 'outputspec.scan',
                             scan_params, 'scan')

            # connect in constants
            scan_params.inputs.set(
                pipeconfig_tr=conf.TR,
                pipeconfig_tpattern=conf.slice_timing_pattern,
                pipeconfig_start_indx=conf.startIdx,
                pipeconfig_stop_indx=conf.stopIdx
            )

            # node to convert TR between seconds and milliseconds
            convert_tr = pe.Node(function.Function(input_names=['tr'],
                                                   output_names=['tr'],
                                                   function=get_tr,
                                                   as_module=True),
                                 name='convert_tr_%d' % str(subject_id))

            # strat.update_resource_pool({
            #     'raw_functional': (func_wf, 'outputspec.rest'),
            #     'scan_id': (func_wf, 'outputspec.scan')
            # })

            trunc_wf = create_wf_edit_func(
                wf_name="edit_func_%d" % str(subject_id)
            )

            # connect the functional data from the leaf node into the wf
            workflow.connect(func_wf, 'outputspec.rest',
                             trunc_wf, 'inputspec.func')

            # connect the other input parameters
            workflow.connect(scan_params, 'start_indx',
                             trunc_wf, 'inputspec.start_idx')
            workflow.connect(scan_params, 'stop_indx',
                             trunc_wf, 'inputspec.stop_idx')

        # replace the leaf node with the output from the recently added
        # workflow
        # strat.set_leaf_properties(trunc_wf, 'outputspec.edited_func')

            # Functional Image Preprocessing Workflow
            if 1 in conf.gen_custom_template:
                meth = 'median'
            else:
                meth = 'mean'

            if (isinstance(conf.functionalMasking, list) and
                    len(conf.functionalMasking) > 1):
                # For now, we just skullstrip using the first method selected
                func_masking = conf.functionalMasking[0]
            else:
                func_masking = conf.functionalMasking

            if func_masking == '3dAutoMask':
                func_preproc = create_func_preproc(
                    use_bet=False,
                    meth=meth,
                    wf_name='func_preproc_automask_%d' % str(subject_id)
                )

                workflow.connect(trunc_wf, meth, 'outputspec.edited_func',
                                 func_preproc, 'inputspec.func')

                func_preproc.inputs.inputspec.twopass = \
                    getattr(conf, 'functional_volreg_twopass', True)

            if func_masking == 'BET':
                func_preproc = create_func_preproc(use_bet=True,
                                                   meth=meth,
                                                   wf_name='func_preproc_bet_%d' % str(subject_id))

                workflow.connect(trunc_wf, meth, 'outputspec.edited_func',
                                 func_preproc, 'inputspec.func')

                func_preproc.inputs.inputspec.twopass = \
                    getattr(conf, 'functional_volreg_twopass', True)

            func_preproc, 'outputspec.preprocessed'

            print("DOOOOOONE")


def anat_workflow(sessions, conf, input_creds_path):
    # TODO ASH temporary code, remove
    # TODO ASH maybe scheme validation/normalization
    already_skullstripped = conf.already_skullstripped[0]
    if already_skullstripped == 2:
        already_skullstripped = 0
    elif already_skullstripped == 3:
        already_skullstripped = 1

    skullstrip_meth = {
        'anatomical_brain_mask': 'mask',
        'BET': 'fsl',
        'AFNI': 'afni'
    }



    subject_id = sessions[0]['subject_id']

    anat_preproc_list = []
    for ses in sessions:

        unique_id = ses['unique_id']
        if 'brain_mask' in ses.keys():
            if ses['brain_mask'] and ses[
                'brain_mask'].lower() != 'none':
                brain_flow = create_anat_datasource(
                    'brain_gather_%d' % unique_id)
                brain_flow.inputs.inputnode.subject = subject_id
                brain_flow.inputs.inputnode.anat = ses['brain_mask']
                brain_flow.inputs.inputnode.creds_path = input_creds_path
                brain_flow.inputs.inputnode.dl_dir = conf.workingDirectory


        # if "AFNI" in conf.skullstrip_option:
        #
        # if "BET" in conf.skullstrip_option:
        #
        # wf = pe.Workflow(name='anat_preproc' + unique_id)
        anat_datasource = create_anat_datasource('anat_gather_%d' % unique_id)
        anat_datasource.inputs.inputnode.subject = subject_id
        anat_datasource.inputs.inputnode.anat = ses['anat']
        anat_datasource.inputs.inputnode.creds_path = input_creds_path
        anat_datasource.inputs.inputnode.dl_dir = conf.workingDirectory

        # anat_prep = create_anat_preproc(skullstrip_meth[])
        # anat_preproc_list.append(wf)

    template_creation_flirt([node.ouputs for node in anat_preproc_list])

    return