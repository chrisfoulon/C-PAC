# -*- coding: utf-8 -*-
import os
import copy
import time
import shutil

from nipype import config
from nipype import logging
import nipype.pipeline.engine as pe
import nipype.interfaces.afni as afni
import nipype.interfaces.io as nio
from nipype.interfaces.utility import Merge
import nipype.interfaces.utility as util

import CPAC
from CPAC.utils.datasource import (
    resolve_resolution,
    create_anat_datasource,
    create_func_datasource,
    create_check_for_s3_node
)

from CPAC.anat_preproc.anat_preproc import create_anat_preproc
from CPAC.func_preproc.func_preproc import (
    create_func_preproc,
    create_wf_edit_func
)
from CPAC.anat_preproc.longitudinal_preproc import subject_specific_template

from CPAC.utils import Strategy, find_files, function

from CPAC.utils.utils import (
    check_config_resources,
    check_system_deps,
    get_scan_params,
    get_tr
)

logger = logging.getLogger('nipype.workflow')


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
    wf_list = []
    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = conf.workingDirectory
    for sub_dict in sub_list:
        if 'func' in sub_dict:
            # for run in sub_dict['func']:
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

            func_paths_dict = sub_dict['func']
            subject_id = sub_dict['subject_id']
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

            func_wf = create_func_datasource(func_paths_dict,
                                             'func_gather_%s' % str(subject_id))
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
                    name='scan_params_%s' % str(subject_id))

            workflow_name = 'resting_preproc_' + str(subject_id)
            workflow = pe.Workflow(name=workflow_name)
            workflow.base_dir = conf.workingDirectory
            workflow.config['execution'] = {
                'hash_method': 'timestamp',
                'crashdump_dir': os.path.abspath(conf.crashLogDirectory)
            }

            if "Selected Functional Volume" in conf.func_reg_input:
                get_func_volume = pe.Node(interface=afni.Calc(),
                                          name='get_func_volume_%s' % str(
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
                                 name='convert_tr_%s' % str(subject_id))

            # strat.update_resource_pool({
            #     'raw_functional': (func_wf, 'outputspec.rest'),
            #     'scan_id': (func_wf, 'outputspec.scan')
            # })

            trunc_wf = create_wf_edit_func(
                wf_name="edit_func_%s" % str(subject_id)
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

            if isinstance(conf.functionalMasking, list):
                # For now, we just skullstrip using the first method selected
                func_masking = conf.functionalMasking[0]
            else:
                func_masking = conf.functionalMasking
            print(str(func_masking))
            if func_masking == '3dAutoMask':
                func_preproc = create_func_preproc(
                    use_bet=False,
                    meth=meth,
                    wf_name='func_preproc_automask_%s' % str(subject_id)
                )

                workflow.connect(trunc_wf, 'outputspec.edited_func',
                                 func_preproc, 'inputspec.func')

                func_preproc.inputs.inputspec.twopass = \
                    getattr(conf, 'functional_volreg_twopass', True)

            elif func_masking == 'BET':
                func_preproc = create_func_preproc(use_bet=True,
                                                   meth=meth,
                                                   wf_name='func_preproc_bet_%s' % str(subject_id))

                workflow.connect(trunc_wf, 'outputspec.edited_func',
                                 func_preproc, 'inputspec.func')

                func_preproc.inputs.inputspec.twopass = \
                    getattr(conf, 'functional_volreg_twopass', True)
            else:
                raise ValueError("functional masking method unsupported: " + str(func_masking))

            # workflow.connect(func_preproc, 'outputspec.preprocessed', datasink,
            #                  'preproc_func')

            wf_list.append(workflow)

            # func_preproc, 'outputspec.preprocessed'

    print("DOOOOOONE")
    return wf_list


def print_out(s):
    print("%%%%%%%%PRINT OUTPUT OF NODE%%%%%%%%")
    print(s)
    print(type(s))
    print("%%%%%%%%PRINT OUTPUT OF NODE%%%%%%%%")
    return s


def anat_longitudinal_workflow(subject_id_dict, conf):
    """

    Parameters
    ----------
    subject_id_dict : dict of list
        Dict of the subjects, the keys are the subject ids and each element
        contains the list of the sessions for each subject.
    conf

    Returns
    -------

    """
    from nipype import config
    config.enable_debug_mode()
    # TODO ASH temporary code, remove
    # TODO ASH maybe scheme validation/normalization
    already_skullstripped = conf.already_skullstripped[0]
    if already_skullstripped == 2:
        already_skullstripped = 0
    elif already_skullstripped == 3:
        already_skullstripped = 1

    template_skull_for_anat_path = resolve_resolution(
        conf.resolution_for_anat,
        conf.template_skull_for_anat,
        'template_skull_for_anat',
        'resolution_for_anat')

    template_center_of_mass = pe.Node(
        interface=afni.CenterMass(),
        name='template_skull_for_anat_center_of_mass'
    )
    template_center_of_mass.inputs.cm_file = os.path.join(
        os.getcwd(), "template_center_of_mass.txt")
    template_center_of_mass.inputs.in_file = template_skull_for_anat_path

    # For each participant we have a list of dict (each dict is a session)
    for subject_id, sub_list in subject_id_dict.items():
        print(str(subject_id))
        workflow = pe.Workflow(
            name="participant_specific_template_" + str(subject_id))
        workflow.base_dir = conf.workingDirectory

        anat_preproc_list_list = []
        # Loop over the sessions to create the input for the longitudinal algo
        for session in sub_list:

            try:
                creds_path = session['creds_path']
                if creds_path and 'none' not in creds_path.lower():
                    if os.path.exists(creds_path):
                        input_creds_path = os.path.abspath(creds_path)
                    else:
                        err_msg = 'Credentials path: "%s" for subject "%s" ' \
                                  'was not found. Check this path and try ' \
                                  'again.' % (creds_path, subject_id)
                        raise Exception(err_msg)
                else:
                    input_creds_path = None
            except KeyError:
                input_creds_path = None

            strat = Strategy()
            strat_list = []

            unique_id = session['unique_id']
            node_suffix = '_'.join([subject_id, unique_id])

            anat_rsc = create_anat_datasource(
                'anat_gather_%s' % node_suffix)
            anat_rsc.inputs.inputnode.subject = subject_id
            anat_rsc.inputs.inputnode.anat = session['anat']
            anat_rsc.inputs.inputnode.creds_path = input_creds_path
            anat_rsc.inputs.inputnode.dl_dir = conf.workingDirectory

            strat.update_resource_pool({
                'anatomical': (anat_rsc, 'outputspec.anat')
            })

            strat.update_resource_pool({
                'template_cmass': (template_center_of_mass, 'cm')
            })

            def connect_anat_preproc_inputs(strat_in, anat_preproc_in):
                new_strat_out = strat_in.fork()
                # print_node = pe.Node(
                #     function.Function(input_names=['s'],
                #                       output_names=['out'],
                #                       function=print_out,
                #                       as_module=True),
                #     name='print_output_' + node_suffix)
                #
                # node, out = strat['template_cmass']
                # workflow.connect(node, out,
                #                  print_node, 's')

                tmp_node, out_key = new_strat_out['anatomical']
                workflow.connect(tmp_node, out_key, anat_preproc_in, 'inputspec.anat')

                workflow.connect(template_center_of_mass, 'cm',
                                 anat_preproc_in, 'inputspec.template_cmass')

                new_strat_out.append_name(anat_preproc_in.name)
                new_strat_out.set_leaf_properties(anat_preproc_in, 'outputspec.brain')
                new_strat_out.update_resource_pool({
                    'anatomical_brain': (
                        anat_preproc_in, 'outputspec.brain'),
                    'anatomical_reorient': (
                        anat_preproc_in, 'outputspec.reorient'),
                })
                return new_strat_out

            """
            Here we don't need a loop for the strategies and there will be no
            other strat before we arrive at this point. 
            In cpac_pipeline, an anatomical image can either have a brain mask 
            or be already skullstripped or have one or several of the skull
            stripping methods. Hence, here we test:
             1) if we have a brain mask, so 
            the skull stripping will just use the mask to remove the skull. 
             2) if the skullstripping has already been done, we just don't do it
             3) else we use the skull stripping methods selected (one or several
             in parallel) 
            """
            if 'brain_mask' in session.keys() and session['brain_mask'] and \
                    session['brain_mask'].lower() != 'none':

                brain_rsc = create_anat_datasource(
                    'brain_gather_%s' % unique_id)
                brain_rsc.inputs.inputnode.subject = subject_id
                brain_rsc.inputs.inputnode.anat = session['brain_mask']
                brain_rsc.inputs.inputnode.creds_path = input_creds_path
                brain_rsc.inputs.inputnode.dl_dir = conf.workingDirectory

                skullstrip_meth = 'mask'
                preproc_wf_name = 'anat_preproc_mask_%s' % node_suffix

                strat.append_name(brain_rsc.name)
                strat.update_resource_pool({
                    'anatomical_brain_mask': (brain_rsc, 'outputspec.anat')
                })

                anat_preproc = create_anat_preproc(
                    method=skullstrip_meth,
                    wf_name=preproc_wf_name,
                    non_local_means_filtering=conf.non_local_means_filtering,
                    n4_correction=conf.n4_bias_field_correction)

                workflow.connect(brain_rsc, 'outputspec.brain_mask',
                                 anat_preproc, 'inputspec.brain_mask')
                new_strat = connect_anat_preproc_inputs(strat, anat_preproc)
                strat_list.append(new_strat)

            elif already_skullstripped:
                skullstrip_meth = None
                preproc_wf_name = 'anat_preproc_already_%s' % node_suffix
                anat_preproc = create_anat_preproc(
                    method=skullstrip_meth,
                    already_skullstripped=True,
                    wf_name=preproc_wf_name,
                    non_local_means_filtering=conf.non_local_means_filtering,
                    n4_correction=conf.n4_bias_field_correction
                )
                new_strat = connect_anat_preproc_inputs(strat, anat_preproc)
                strat_list.append(new_strat)

            else:

                if "AFNI" in conf.skullstrip_option:
                    skullstrip_meth = 'afni'
                    preproc_wf_name = 'anat_preproc_afni_%s' % node_suffix

                    anat_preproc = create_anat_preproc(
                        method=skullstrip_meth,
                        wf_name=preproc_wf_name,
                        non_local_means_filtering=conf.non_local_means_filtering,
                        n4_correction=conf.n4_bias_field_correction)

                    anat_preproc.inputs.AFNI_options.set(
                        shrink_factor=conf.skullstrip_shrink_factor,
                        var_shrink_fac=conf.skullstrip_var_shrink_fac,
                        shrink_fac_bot_lim=conf.skullstrip_shrink_factor_bot_lim,
                        avoid_vent=conf.skullstrip_avoid_vent,
                        niter=conf.skullstrip_n_iterations,
                        pushout=conf.skullstrip_pushout,
                        touchup=conf.skullstrip_touchup,
                        fill_hole=conf.skullstrip_fill_hole,
                        avoid_eyes=conf.skullstrip_avoid_eyes,
                        use_edge=conf.skullstrip_use_edge,
                        exp_frac=conf.skullstrip_exp_frac,
                        smooth_final=conf.skullstrip_smooth_final,
                        push_to_edge=conf.skullstrip_push_to_edge,
                        use_skull=conf.skullstrip_use_skull,
                        perc_int=conf.skullstrip_perc_int,
                        max_inter_iter=conf.skullstrip_max_inter_iter,
                        blur_fwhm=conf.skullstrip_blur_fwhm,
                        fac=conf.skullstrip_fac,
                        monkey=conf.skullstrip_monkey,
                    )

                    new_strat = connect_anat_preproc_inputs(strat, anat_preproc)
                    strat_list.append(new_strat)

                if "BET" in conf.skullstrip_option:
                    skullstrip_meth = 'fsl'
                    preproc_wf_name = 'anat_preproc_fsl_%s' % node_suffix

                    anat_preproc = create_anat_preproc(
                        method=skullstrip_meth,
                        wf_name=preproc_wf_name,
                        non_local_means_filtering=conf.non_local_means_filtering,
                        n4_correction=conf.n4_bias_field_correction)

                    anat_preproc.inputs.BET_options.set(
                        frac=conf.bet_frac,
                        mask_boolean=conf.bet_mask_boolean,
                        mesh_boolean=conf.bet_mesh_boolean,
                        outline=conf.bet_outline,
                        padding=conf.bet_padding,
                        radius=conf.bet_radius,
                        reduce_bias=conf.bet_reduce_bias,
                        remove_eyes=conf.bet_remove_eyes,
                        robust=conf.bet_robust,
                        skull=conf.bet_skull,
                        surfaces=conf.bet_surfaces,
                        threshold=conf.bet_threshold,
                        vertical_gradient=conf.bet_vertical_gradient,
                    )

                    new_strat = connect_anat_preproc_inputs(strat, anat_preproc)
                    strat_list.append(new_strat)

                if not any(o in conf.skullstrip_option for o in
                           ["AFNI", "BET"]):
                    err = '\n\n[!] C-PAC says: Your skull-stripping ' \
                          'method options setting does not include either' \
                          ' \'AFNI\' or \'BET\'.\n\n Options you ' \
                          'provided:\nskullstrip_option: {0}\n\n'.format(
                            str(conf.skullstrip_option))
                    raise Exception(err)

            # Here we have at least new_strat in strat_list

            if len(anat_preproc_list_list) == 0:
                # just initialize the list of list if it is empty
                anat_preproc_list_list = [[] for _ in strat_list]

            for num_strat, loop_strat in enumerate(strat_list):

                node, out_file = loop_strat['anatomical_brain']

                anat_preproc_list_list[num_strat].append(node)

        for anat_preproc_list in anat_preproc_list_list:

            merge_node = pe.Node(
                interface=Merge(len(anat_preproc_list)),
                name="anat_longitudinal_merge")

            # the in{}.format take i+1 because the Merge nodes inputs starts at 1...
            for i in range(len(anat_preproc_list)):
                workflow.connect(anat_preproc_list[i],
                                 'outputspec.brain', merge_node,
                                 'in{}'.format(i + 1))

            template_node = subject_specific_template(
                workflow_name='_'.join(['subject_specific_template', subject_id])
            )

            workflow.connect(merge_node, 'out', template_node, 'img_list')
            template_node.inputs.output_folder = os.getcwd()
            template_node.inputs.set(
                avg_method=conf.long_reg_avg_method,
                dof=conf.dof,
                interp=conf.interp,
                cost=conf.cost,
                convergence_threshold=conf.convergence_threshold,
                thread_pool=conf.thread_pool,
            )
            workflow.run()

    return

